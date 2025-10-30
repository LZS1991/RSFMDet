# RSFMDet: Remote Sensing Feature Mining Detector

RSFMDet is a remote sensing object detection framework based on MMDetection, designed for efficient and accurate detection in aerial and satellite imagery.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Model Configurations](#model-configurations)
- [Performance](#performance)
- [License](#license)
- [Citation](#citation)

## Introduction

RSFMDet (Remote Sensing Feature Mining Detector) is a specialized object detection framework tailored for remote sensing applications. Built upon the robust MMDetection framework, RSFMDet incorporates efficient vision transformers and lightweight architectures to achieve high performance on aerial and satellite imagery while maintaining computational efficiency.

## Features

- Multiple state-of-the-art detection architectures
- Optimized for remote sensing datasets (DIOR, DOTA, UCAS-AOD, xView)
- Support for various backbones including EfficientViT, ConvNeXt, PVT, MobileNet, etc.
- Auxiliary learning modules for enhanced feature extraction
- Comprehensive configuration files for different datasets and models
- Easy training and evaluation pipeline

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd RSFMDet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install MMDetection (if not already installed):
```bash
pip install mmdet
```

For detailed installation instructions, please refer to the [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## Dataset Preparation

RSFMDet supports multiple remote sensing datasets:

- DIOR (Diversified-Image Object Recognition)
- DOTA (Dataset for Object deTection in Aerial images)
- UCAS-AOD (University of California, San Diego Aerial Object Detection)
- xView

Please organize your datasets according to the MMDetection format. Example configurations can be found in the [configs](configs/) directory.

## Training

To train a model with a specific configuration:

```bash
python tools/train.py configs/<config-file>.py
```

For distributed training:
```bash
bash tools/dist_train.sh configs/<config-file>.py <num-gpus>
```

## Testing

To evaluate a trained model:

```bash
python tools/test.py configs/<config-file>.py work_dirs/<checkpoint>.pth
```

For distributed testing:
```bash
bash tools/dist_test.sh configs/<config-file>.py work_dirs/<checkpoint>.pth <num-gpus>
```

## Model Configurations

RSFMDet provides configurations for various architectures and datasets:

### DIOR Dataset
Configurations for the DIOR dataset can be found in [configs/_lzs2851_dior/](configs/_lzs2851_dior/)

### DOTA Dataset
Configurations for the DOTA dataset can be found in [configs/_lzs2851_dota/](configs/_lzs2851_dota/)

### UCAS-AOD Dataset
Configurations for the UCAS-AOD dataset can be found in [configs/_lzs2851_ucas/](configs/_lzs2851_ucas/)

### xWheel Dataset
Configurations for the xWheel dataset can be found in [configs/_lzs2851_xwheel/](configs/_lzs2851_xwheel/)

Supported architectures include:
- RetinaNet with various backbones (EfficientViT, ConvNeXt, PVT, MobileNet, etc.)
- Mask R-CNN variants
- SSD-Lite models

## Performance

Performance benchmarks for various models on different datasets:

| Model | Backbone | Dataset | mAP | FPS |
|-------|----------|---------|-----|-----|
| RetinaNet | PS-LT | DIOR | - | - |
| RetinaNet | PS-LT-B | DIOR | - | - |
| RetinaNet | ConvNeXtV2-Atto | DIOR | - | - |
| RetinaNet | PVTv2-B0 | DIOR | - | - |

*(Detailed performance metrics will be added based on experimental results)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find RSFMDet useful in your research, please consider citing:

```bibtex
@article{rsfmdet2022,
  title={RSFMDet: Remote Sensing Feature Mining Detector},
  author={<Author Names>},
  journal={<Journal Name>},
  year={2022}
}
```

## Acknowledgements

This project is built upon the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework. We thank the MMDetection team for their great work.