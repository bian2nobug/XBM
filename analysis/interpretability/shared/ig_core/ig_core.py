"""
===============================================================================
IG Core - Integrated Gradients core module
===============================================================================

[Purpose]
    Compute Integrated Gradients attribution for any PyTorch model

[Theory]
    - Paper: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
    - Formula: IG(x) = (x - baseline) x integral of dF(interpolated)/dx over alpha
    - Axioms: Sensitivity, Implementation Invariance, Completeness

[Core classes]
    - IntegratedGradients: IG algorithm implementation, supports arbitrary input shapes
    - BaseAttributionReducer: abstract base for attribution reduction; subclass to customize
    - IGAnalyzer: analyzer that integrates IG computation and attribution reduction

[Usage]
    >>> from ig_core import IGAnalyzer
    >>> analyzer = IGAnalyzer(model, n_steps=50, device='cuda:0')
    >>> results = analyzer.analyze(inputs, target=0)

[Extension]
    Subclass BaseAttributionReducer to implement custom reduction -> see aggregators.py
===============================================================================
"""

import gc
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Dict, Any, List
from tqdm import tqdm


class IntegratedGradients:
    """
    General-purpose Integrated Gradients attribution analyzer
    
    Supports any PyTorch model and input tensors of arbitrary shape.
    
    Reference:
        Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
    
    Args:
        model: a trained PyTorch model
        n_steps: number of integration steps; larger is more precise (default 50, recommended 50-300)
        device: compute device ('cpu', 'cuda:0', etc.)
        
    Example:
        >>> ig = IntegratedGradients(model, n_steps=50, device='cuda:0')
        >>> attributions = ig.attribute(inputs, target=0)
    """
    
    def __init__(self, model: nn.Module, n_steps: int = 50, device: str = 'cpu'):
        self.model = model
        self.n_steps = n_steps
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def _get_target_score(self,
                          outputs: torch.Tensor,
                          target: Optional[Union[int, torch.Tensor, Callable]] = None
                          ) -> torch.Tensor:
        """
        Extract the target score from the model outputs

        Args:
            outputs: model output tensor
            target: target class/index/custom function
                - None: use the predicted class (argmax)
                - int: a specified class index (for binary: 0 or 1)
                - Tensor: per-sample class index
                - Callable: custom function f(outputs) -> scalar

        Returns:
            scalar: a scalar value used for backpropagation
        """
        if target is None:
            # for binary classification, decide by the sigmoid threshold
            if outputs.shape[-1] == 1:
                target = (outputs.sigmoid() > 0.5).long().squeeze(-1)
            else:
                target = outputs.argmax(dim=-1)

        if callable(target):
            return target(outputs)

        if isinstance(target, int):
            target = torch.tensor([target] * outputs.shape[0], device=self.device)

        if outputs.dim() == 1:
            return outputs.sum()

        # binary case: output dim is 1, using BCEWithLogitsLoss
        # target=1: maximize the output (positive class)
        # target=0: minimize the output (negative class) -> negate
        if outputs.shape[-1] == 1:
            # convert target to a tensor
            if not isinstance(target, torch.Tensor):
                target = torch.tensor([target], device=self.device)
            # coefficient is 1 when target=1, -1 when target=0
            sign = 2 * target.float() - 1  # 0->-1, 1->1
            return (outputs.squeeze(-1) * sign).sum()
        
        return outputs.gather(1, target.view(-1, 1)).sum()
    
    def _compute_gradients(self, 
                           inputs: torch.Tensor, 
                           target: Optional[Union[int, torch.Tensor, Callable]] = None
                           ) -> torch.Tensor:
        """
        Compute the gradient of the target score with respect to the inputs
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        
        self.model.zero_grad()
        outputs = self.model(inputs)
        
        target_score = self._get_target_score(outputs, target)
        target_score.backward()
        
        return inputs.grad.clone()
    
    def attribute(self,
                  inputs: torch.Tensor,
                  baseline: Optional[torch.Tensor] = None,
                  target: Optional[Union[int, torch.Tensor, Callable]] = None,
                  return_convergence_delta: bool = False,
                  internal_batch_size: int = 10
                  ) -> Union[torch.Tensor, tuple]:
        """
        Compute Integrated Gradients attribution

        IG formula: IG(x) = (x - baseline) x integral over [0,1] of dF(baseline + alpha*(x-baseline))/dx d(alpha)
        """
        inputs = inputs.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        baseline = baseline.to(self.device)
        
        total_gradients = torch.zeros_like(inputs)
        
        for start in range(0, self.n_steps, internal_batch_size):
            end = min(start + internal_batch_size, self.n_steps)
            
            for i in range(start + 1, end + 1):
                alpha = float(i) / self.n_steps
                scaled_input = baseline + alpha * (inputs - baseline)
                grad = self._compute_gradients(scaled_input, target)
                total_gradients += grad
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_gradients = total_gradients / self.n_steps
        attributions = (inputs - baseline) * avg_gradients
        
        if return_convergence_delta:
            delta = self._compute_convergence_delta(inputs, baseline, attributions, target)
            return attributions, delta

        return attributions

    def _compute_convergence_delta(self,
                                   inputs: torch.Tensor,
                                   baseline: torch.Tensor,
                                   attributions: torch.Tensor,
                                   target: Optional[Union[int, torch.Tensor, Callable]]
                                   ) -> torch.Tensor:
        """Compute the IG convergence delta (Completeness axiom check)"""
        with torch.no_grad():
            output_input = self.model(inputs)
            output_baseline = self.model(baseline)

            score_input = self._get_target_score(output_input, target)
            score_baseline = self._get_target_score(output_baseline, target)

            expected_diff = score_input - score_baseline
            actual_sum = attributions.sum()

            delta = (expected_diff - actual_sum).abs()

        return delta


class BaseAttributionReducer(ABC):
    """Base class for attribution reducers"""

    @abstractmethod
    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    def get_top_k(self, attribution: torch.Tensor, k: int = 10, dim: int = -1) -> tuple:
        values, indices = torch.topk(attribution.abs(), k, dim=dim)
        return indices, values


class SimpleReducer(BaseAttributionReducer):
    """Simple attribution reducer"""

    def __init__(self, reduce_dims: Optional[List[int]] = None):
        self.reduce_dims = reduce_dims

    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {'full_attribution': attributions}
        if self.reduce_dims is not None:
            results['reduced'] = attributions.sum(dim=self.reduce_dims)
        results['total'] = attributions.sum()
        results['mean'] = attributions.mean()
        results['abs_sum'] = attributions.abs().sum()
        return results


class IGAnalyzer:
    """IG analyzer: integrates IG computation and attribution reduction"""

    def __init__(self,
                 model: nn.Module,
                 reducer: Optional[BaseAttributionReducer] = None,
                 n_steps: int = 50,
                 device: str = 'cpu'):
        self.ig = IntegratedGradients(model, n_steps, device)
        self.reducer = reducer or SimpleReducer()
        self.device = device

    def analyze(self,
                inputs: torch.Tensor,
                baseline: Optional[torch.Tensor] = None,
                target: Optional[Union[int, torch.Tensor, Callable]] = None,
                return_raw: bool = False
                ) -> Dict[str, Any]:
        attributions = self.ig.attribute(inputs, baseline, target)
        results = self.reducer.reduce(attributions)

        if not return_raw and 'full_attribution' in results:
            del results['full_attribution']

        return results


def create_baseline(inputs: torch.Tensor, method: str = 'zero', **kwargs) -> torch.Tensor:
    """Utility function to create a baseline"""
    if method == 'zero':
        return torch.zeros_like(inputs)
    elif method == 'mean':
        return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
    elif method == 'random':
        return torch.randn_like(inputs) * kwargs.get('std', 0.1)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
