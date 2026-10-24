# DMBNet

Official Pytorch Code base for "A DINOv3-Driven Framework with Multiscale Adaptation and Boundary Refinement for Martian Slope Streak Change Detection" [[Paper](https://ieeexplore.ieee.org/document/11737601)]

## Introduction

We propose a novel pixel-level slope streak change detection approach, named DMBNet. It aims to extract fine change streak information by integrating a VFM (i.e., DINOv3) with boundary refinement and multiscale adaptation. To the best of our knowledge, this is the first work to detect pixel-level slope streak changes using a tailored deep learning network.


## Using the code:

The code is stable while using Python 3.9.22, CUDA >=12.1

- Clone this repository:
```bash
git clone https://github.com/long123524/DMBNet
cd /DMBNet
```

To install all the dependencies using conda or pip:

```bash
pip install -r requirements.txt
```

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── A
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── B
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── label
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── boundary
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    
```

For testing and validation datasets, the same structure as the above.

## Datasets

We are further expanding the Martian slope streak change detection dataset, which will be available for use.

## Training

python train.py 

## Test
python prediction.py

## Evaluation

python eval_CD.py

## A pretrained weight
A pretrained weight of PVT-V2 on the ImageNet dataset is provided: https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing
A pretrained weight of DINOv3 on the LVD dataset is provided: https://drive.google.com/file/d/1q4OoQAyxFd6zZmWQlqIXqV2KFuiVDxzZ/view?usp=sharing

### Citation:
If you find this work useful or interesting, please consider citing the following references.
```
@article{long2026dinov3,
  title={A DINOv3-Driven Framework with Multiscale Adaptation and Boundary Refinement for Martian Slope Streak Change Detection},
  author={Long, Jiang and Liu, Sicong and Wan, Bo and Xu, Yusheng and Xie, Huan and Tong, Xiaohua},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={23},
  pages={6022205-6022205},
  year={2026},
  publisher={IEEE}
}
```
