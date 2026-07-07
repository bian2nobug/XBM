# Third-Party Notices

This repository contains the implementation of XBM and related analysis utilities for research use in computational pathology.

Several external methods, repositories, pretrained models, and software tools inspired or supported this work. They are acknowledged below for transparency. Unless explicitly stated otherwise, the source code in this repository is released under the MIT License.

## Referenced methods and software

### CLAM

This project uses a CLAM-inspired whole-slide image preprocessing workflow, including tissue-region detection, tile extraction, and weakly supervised computational pathology conventions.

Users should consult the original CLAM repository and publication for its license terms, usage conditions, and citation requirements.

### TransMIL

This project includes TransMIL-style baseline and reference model definitions for weakly supervised WSI modeling.

Users should consult the original TransMIL repository and publication for its license terms, usage conditions, and citation requirements.

### CS-MIL and multi-scale pathology modeling

This project includes multi-scale fusion components and model variants inspired by cross-scale multi-instance learning and related multi-resolution computational pathology methods.

Users should consult the corresponding original publications and repositories for their license terms and citation requirements.

### Prov-GigaPath

This project assumes tile-level pathology features can be extracted using pretrained pathology foundation models such as Prov-GigaPath.

Users are responsible for complying with the license, model-use policy, and data-use restrictions of any pretrained model used together with this repository.

### HistomicsTK

This project refers to HistomicsTK-based color normalization utilities for H&E stain normalization.

Users should consult the HistomicsTK project for its license terms and citation requirements.

### OpenSlide and WSI tooling

This project may be used together with OpenSlide and related whole-slide image processing tools.

Users should consult the corresponding software licenses and installation requirements.

## License compatibility note

The MIT License in this repository applies to code authored for this project unless otherwise noted.

If users replace, extend, or combine this repository with third-party code, pretrained models, or datasets, they are responsible for complying with all applicable third-party licenses and terms of use.

If any file is directly derived from third-party source code with a different license, the original license terms for that file should be retained and respected.

## Research-use notice

This repository is intended for research and educational use. It is not intended for clinical diagnosis, treatment selection, or direct medical decision-making.
