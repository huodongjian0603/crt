# CRT
Official PyTorch implementation for our paper "Dual Progressive Transformations for Weakly Supervised Semantic Segmentation"
## Abstract
Weakly supervised semantic segmentation (WSSS), which aims to mine the object regions by merely using class-level labels, is a challenging task in computer vision. The current state-of-the-art CNN-based methods usually adopt Class-Activation-Maps (CAMs) to highlight the potential areas of the object, however, they may suffer from the part-activated issues. To this end, we try an early attempt to explore the global feature attention mechanism of vision transformer in WSSS task. However, since the transformer lacks the inductive bias as in CNN models, it can not boost the performance directly and may yield the over-activated problems. To tackle these drawbacks, we propose a Convolutional Neural Networks Refined Transformer (CRT) to mine a globally complete and locally accurate class activation maps in this paper. To validate the effectiveness of our proposed method, extensive experiments are conducted on PASCAL VOC 2012 and CUB-200-2011 datasets. Experimental evaluations show that our proposed CRT achieves the new state-of-the-art performance on both the weakly supervised semantic segmentation task the weakly supervised object localization task, which outperform others by a large margin.
## Requirement
* Python 3.7
* PyTorch 1.1.0+
* NVIDIA GeForce RTX 2080Ti x 2
## Usage
### Preparation
1. Download the repository.
```
git clone https://github.com/huodongjian0603/crt.git
```
2. Install dependencies.
```
pip install -r requirements.txt
```
3. Download [PASCAL VOC 2012 devkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
### Generate pseudo-segmentation labels
1. Run script `run_sample.py`.
```
python run_sample --voc12_root $downloaded_dataset_path/VOCdevkit/VOC2012
```
After the script completes, pseudo labels are generated in the following directory and their quality is evaluated in mIoU.
```
.
├── misc
├── net
├── result  # generated cam and pseudo labels
│   ├── cam
│   ├── ins_seg
│   ├── ir_label
│   └── sem_seg # what we want in this step!
├── sess  # saved models
│   ├── deits_cam.pth
│   ├── res152_cam.pth
│   └── res50_irn.pth
├── step
├── voc12
├── requirements.txt
├── run_sample.py
└── sample_train_eval.log
```
### Train DeepLab with the generated pseudo labels.

## TODO
link of paper, picture of crt，performance(quality and quantity), deeplab stage tutorial.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```latex
@misc{2203.04708,
Author = {xxx},
Title = {xxx},
Year = {2022},
Eprint = {arXiv:2203.04708},
}

```


## Acknowledgement

Our project references the codes in the following repos.

- [IRNet](https://github.com/jiwoon-ahn/irn), [TSCAM](https://github.com/vasgaowei/TS-CAM) and [L2G](https://github.com/PengtaoJiang/L2G)

