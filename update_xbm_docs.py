from pathlib import Path

MODEL_INVENTORY = """# Model Inventory

This document summarizes the model definitions included in this repository.

The main XBM implementation is retained in `model/XBM.py`. Additional model files are provided as architecture definitions and component references for baseline or ablation-style model variants. This repository does not provide a complete end-to-end reproduction pipeline for all ablation experiments.

## Main model

- `model/XBM.py`  
  Main XBM model entry.

## Baseline models

- `model/multi_attnmil.py`  
  Attention-based MIL baseline.

- `model/multi_transmil_multiscale_multimodal.py`  
  TransMIL-style multi-scale / multimodal baseline.

- `model/baselines/transmil_lite.py`  
  Lightweight TransMIL-style baseline component.

## Ablation and reference models

- `model/ablations/clinical_only.py`  
  Clinical-only model definition.

- `model/ablations/pathology_only.py`  
  Pathology-only model definition.

- `model/ablations/single_scale.py`  
  Single-scale / no-multiscale reference model definition.

These files are included to make the architectural components used in model comparison and ablation analysis transparent. They are not intended to reproduce all experimental tables automatically.

## Additional reusable components

- `model/components_extra/coam_fusion.py`  
  COAM / pyramid-style multi-scale fusion component.

- `model/components_extra/transformer.py`  
  Standard Transformer encoder component.

- `model/components_extra/attention.py`  
  Standard multi-head attention component.

- `model/components_extra/feedforward.py`  
  Standard feed-forward component.

- `model/components_extra/mlp_encoder.py`  
  MIL-style MLP encoder.

- `model/components_extra/gated_attention.py`  
  MIL-style gated attention module.

## Original submodules

The original modules under `model/submodel/` are retained for compatibility with the main model and interpretability utilities. These files are not reorganized to avoid breaking checkpoint loading or existing import paths.

## Interpretability adapter

The interpretability code under `analysis/interpretability/shared/model_lib/` contains graph-preserving adapter modules used by the attribution and heatmap scripts. These modules are kept separate from the main model implementation to preserve the tested interpretability workflow.
"""

INTENDED_USE = """## Intended Use and License Notice

This repository is intended for research use in computational pathology and biomedical machine learning. It is not intended for clinical diagnosis, treatment decision-making, or deployment as a medical device.

The source code in this repository is released under the MIT License unless otherwise noted. Third-party methods, repositories, pretrained models, datasets, and software tools that inspired or supported this work are acknowledged in `THIRD_PARTY_NOTICES.md`.

Users are responsible for complying with the licenses and terms of use of any third-party software, pretrained models, datasets, or external tools used together with this repository.
"""

repo = Path.cwd()

if not (repo / ".git").exists():
    raise SystemExit("Error: please run this script from the root of the XBM Git repository.")

model_dir = repo / "model"
model_dir.mkdir(exist_ok=True)

(model_dir / "MODEL_INVENTORY.md").write_text(MODEL_INVENTORY, encoding="utf-8")

readme_changes = model_dir / "README_changes.md"
if readme_changes.exists():
    readme_changes.unlink()

readme = repo / "README.md"
if not readme.exists():
    raise SystemExit("Error: README.md not found.")

text = readme.read_text(encoding="utf-8")

if "## Intended Use and License Notice" not in text:
    markers = ["## Citation", "## License", "## Acknowledgements", "## Acknowledgments"]
    insert_pos = None
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            insert_pos = idx
            break

    if insert_pos is None:
        text = text.rstrip() + "\n\n" + INTENDED_USE + "\n"
    else:
        text = text[:insert_pos].rstrip() + "\n\n" + INTENDED_USE + "\n\n" + text[insert_pos:].lstrip()

    readme.write_text(text, encoding="utf-8")
    print("Updated README.md: added Intended Use and License Notice.")
else:
    print("README.md already contains Intended Use and License Notice; skipped insertion.")

print("Updated model/MODEL_INVENTORY.md.")
print("Deleted model/README_changes.md if it existed.")
