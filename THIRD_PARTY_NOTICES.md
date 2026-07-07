# Third-Party Notices and References

This repository contains original XBM code together with lightweight adapters, wrappers, and analysis utilities that rely on or are inspired by established open-source software and published computational pathology methods. This file is intended to help users identify relevant upstream projects and references. It is not a substitute for the license files or citation requirements of the corresponding third-party resources.

## Software dependencies

The code uses common scientific Python and pathology-processing packages, including PyTorch, torchvision, NumPy, pandas, scikit-learn, h5py, OpenSlide, OpenCV, Pillow, HistomicsTK, timm, optuna, python-igraph, leidenalg, scipy, matplotlib, and umap-learn. Users should follow the licenses and citation instructions of these upstream projects when redistributing or adapting the code.

The R analysis scripts use readr, dplyr, tidyr, and ggplot2.

## Pathology preprocessing and WSI utilities

Parts of the preprocessing workflow follow the common computational pathology pattern of tissue-region detection, tile-coordinate extraction, quality control, HDF5 storage, and WSI heatmap rendering. The implementation is designed to be compatible with CLAM-style WSI preprocessing conventions and OpenSlide-based slide access. Users should cite the relevant CLAM and OpenSlide resources when these tools or conventions are used in downstream work.

## Feature extraction and pathology foundation models

The repository provides code paths for Prov-GigaPath-style tile feature extraction but does not redistribute Prov-GigaPath weights. Users must obtain foundation-model weights from the original source and follow the corresponding model license and citation requirements.

## Model components and methodological references

Several model components are implemented for architectural reference or ablation-model definition, including attention-based multiple-instance learning, Transformer/TransMIL-style WSI modeling, same-FOV multiscale fusion, CS-MIL-inspired fusion, COAM/Pyramid-style fusion, Integrated Gradients attribution, and attention-based heatmap visualization. When using these components, users should cite the corresponding original methodological papers where appropriate.

Relevant method families include, but are not limited to:

- Attention-based multiple-instance learning.
- CLAM-style weakly supervised computational pathology workflows.
- TransMIL and Transformer-based WSI representation learning.
- Cross-scale multi-instance learning and related same-FOV multiscale pathology models.
- Integrated Gradients attribution.
- Prov-GigaPath or other pathology foundation-model feature extractors used to generate tile embeddings.

## Data, pretrained weights, and trained checkpoints

This repository does not redistribute TCGA, CPTAC, WSIs, molecular labels, clinical tables, pretrained foundation-model weights, or trained XBM checkpoints. Users are responsible for obtaining any required datasets and pretrained weights from their original sources and for complying with the associated data-use agreements, licenses, and citation requirements.

## User responsibility

Before public redistribution or reuse in a derived project, users should verify the licenses of all third-party packages, pretrained weights, datasets, and copied or adapted source files used in their local workflow.
