from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAttentionPooling(nn.Module):
    """
    Attention pooling over the scale/view dimension.
    Input:  x (B, C, M, N)
    Output: y (B, C, M)
    """
    def __init__(self, dim: int = 1536) -> None:
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,C,M,N), got shape={tuple(x.shape)}")
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        score = self.scorer(x_perm)
        weight = F.softmax(score, dim=2)
        fused = (weight * x_perm).sum(dim=2).permute(0, 2, 1).contiguous()
        if return_attention:
            return fused, weight.squeeze(-1)
        return fused


# Backward-compatible alias for old training scripts.
FC_AttnPool_Fusion = ScaleAttentionPooling


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    n_latent = n_latent or (n_in + 1) // 2
    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1)
    )


class AdaptableMIL(nn.Module):
    """
    Ilse-style attention MIL aggregator.
    Input:  bags (B, M, n_feats)
    Output: logits (B, n_out)
    """
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        encoder_params: Optional[dict] = None,
        attention_params: Optional[dict] = None,
        head_params: Optional[dict] = None,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        use_layernorm_head: bool = True
    ) -> None:
        super().__init__()
        if encoder_params is None:
            self.encoder = nn.Sequential(nn.Linear(n_feats, hidden_dim), nn.ReLU())
            enc_dim = hidden_dim
        else:
            if encoder is None:
                raise ValueError("encoder must be provided when encoder_params is not None.")
            self.encoder = encoder(**encoder_params)
            enc_dim = encoder_params.get("dim", hidden_dim)
        if attention_params is None:
            self.attention = Attention(enc_dim)
        else:
            if attention is None:
                raise ValueError("attention must be provided when attention_params is not None.")
            self.attention = attention(**attention_params)
        if head_params is None:
            norm = nn.LayerNorm(enc_dim) if use_layernorm_head else nn.BatchNorm1d(enc_dim)
            self.head = nn.Sequential(nn.Flatten(), norm, nn.Dropout(dropout), nn.Linear(enc_dim, n_out))
        else:
            if head is None:
                raise ValueError("head must be provided when head_params is not None.")
            self.head = head(**head_params)

    def forward(self, bags: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, return_attention: bool = False):
        if bags.ndim != 3:
            raise ValueError(f"Expected bags with shape (B,M,C), got {tuple(bags.shape)}")
        if valid_mask is None:
            valid_mask = (bags != 0).any(dim=-1)
        if valid_mask.shape != bags.shape[:2]:
            raise ValueError(f"valid_mask shape must be {tuple(bags.shape[:2])}, got {tuple(valid_mask.shape)}")
        embeddings = self.encoder(bags)
        attention_weights = self._masked_attention_scores(embeddings, valid_mask)
        pooled = (attention_weights * embeddings).sum(dim=1)
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        return logits

    def _masked_attention_scores(self, embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(embeddings)
        mask = valid_mask.to(dtype=torch.bool, device=attention_scores.device).unsqueeze(-1)
        masked_attention = torch.where(mask, attention_scores, torch.full_like(attention_scores, -1e10))
        return torch.softmax(masked_attention, dim=1)


class MultiScaleMultiModalMIL(nn.Module):
    """
    Attention MIL baseline with optional multi-scale fusion and optional auxiliary token.
    Expected input: x (B, n_feats + aux_dim, M, N)
    """
    def __init__(
        self,
        n_feats: int = 1536,
        n_out: int = 1,
        use_multiscale: bool = True,
        view_index: int = 0,
        use_multimodal: bool = True,
        add_token: bool = True,
        aux_hidden: int = 256,
        mil_hidden: int = 256,
        dropout: float = 0.25,
        use_layernorm_head: bool = True
    ) -> None:
        super().__init__()
        self.n_feats = n_feats
        self.use_multiscale = use_multiscale
        self.view_index = view_index
        self.use_multimodal = use_multimodal
        self.add_token = add_token
        self.ms_fusion = ScaleAttentionPooling(dim=n_feats)
        self.mil = AdaptableMIL(n_feats=n_feats, n_out=n_out, hidden_dim=mil_hidden, dropout=dropout, use_layernorm_head=use_layernorm_head)
        self.aux_proj = nn.Sequential(nn.LazyLinear(aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, n_feats), nn.ReLU())

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_scale_attention: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,M,N), got {tuple(x.shape)}")
        B, C, M, N = x.shape
        if C < self.n_feats:
            raise ValueError(f"Input channel dim={C} is smaller than n_feats={self.n_feats}")
        histo = x[:, :self.n_feats, :, :].contiguous()
        scale_attention = None
        if self.use_multiscale:
            if return_scale_attention:
                fused, scale_attention = self.ms_fusion(histo, return_attention=True)
            else:
                fused = self.ms_fusion(histo)
        else:
            if not (0 <= self.view_index < N):
                raise ValueError(f"view_index={self.view_index} is out of range for N={N}")
            fused = histo[..., self.view_index]
        bag = fused.permute(0, 2, 1).contiguous()
        valid_mask = (bag != 0).any(dim=-1)
        if self.use_multimodal and self.add_token and C > self.n_feats:
            aux_vec = x[:, self.n_feats:, 0, 0].contiguous()
            aux_token = self.aux_proj(aux_vec).unsqueeze(1)
            bag = torch.cat([bag, aux_token], dim=1)
            aux_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            valid_mask = torch.cat([valid_mask, aux_mask], dim=1)
        mil_out = self.mil(bag, valid_mask=valid_mask, return_attention=return_attention)
        if return_attention:
            logits, attention = mil_out
            output = {"logits": logits, "attention": attention}
            if return_scale_attention:
                output["scale_attention"] = scale_attention
            return output
        if return_scale_attention:
            return {"logits": mil_out, "scale_attention": scale_attention}
        return mil_out


# Backward-compatible aliases.
AttentionMIL = AdaptableMIL
MultiScaleAttentionMIL = MultiScaleMultiModalMIL


if __name__ == "__main__":
    torch.manual_seed(0)
    B, M, N = 2, 128, 21
    n_feats, aux_dim = 1536, 57
    x = torch.randn(B, n_feats + aux_dim, M, N)
    model = MultiScaleMultiModalMIL(n_feats=n_feats, n_out=4, use_multiscale=True, use_multimodal=True, add_token=True)
    out = model(x, return_attention=True, return_scale_attention=True)
    print(out["logits"].shape, out["attention"].shape, out["scale_attention"].shape)
