# RSFMDet: Foundation Model-based Auxiliary Framework for Object Detection in Aerial Remote Sensing Images


## Table of Contents

- [RSFMDet: Foundation Model-based Auxiliary Framework for Object Detection in Aerial Remote Sensing Images](#rsfmdet-foundation-model-based-auxiliary-framework-for-object-detection-in-aerial-remote-sensing-images)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Training](#training)
  - [Testing](#testing)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Introduction

RSFMDet is a foundation model-based auxiliary framework for object detectionbased on MMDetection, designed for efficient and accurate detection in aerial and satellite imagery.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/LZS1991/RSFMDet.git
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

## Training

To train a model with a specific configuration:

```bash
python tools/train.py configs/_lzs2851_dior/retinanet_convnextv2_atto_fpn_3x_coco.py
```

For distributed training:
```bash
bash tools/dist_train.sh configs/retinanet_convnextv2_atto_fpn_3x_coco.py 4
```

## Testing

To evaluate a trained model:

```bash
python tools/test.py configs/_lzs2851_dior/retinanet_convnextv2_atto_fpn_3x_coco.py work_dirs/epoch_50.pth
```

For distributed testing:
```bash
bash tools/dist_test.sh configs/_lzs2851_dior/retinanet_convnextv2_atto_fpn_3x_coco.py work_dirs/epoch_50.pth 4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find RSFMDet useful in your research, please consider citing:

```bibtex
@article{rsfmdet2025,
  title={Foundation Model-based Auxiliary Framework for Object Detection in Aerial Remote Sensing Images},
  author={Wanjie Lu, Chaoyang Niu, Wei Liu, Tao Hu, Chaozhen Lan, and Shiju Wang},
  journal={<Journal Name>},
  year={2025}
}
```

## Acknowledgements

This project is built upon the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework. We thank the MMDetection team for their great work.