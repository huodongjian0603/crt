# CRT
Official PyTorch implementation for our paper "Dual Progressive Transformations for Weakly Supervised Semantic Segmentation"

<div align="center">
  <img src="fig\outline.png" width="800px">
</div>

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
python run_sample.py --voc12_root $downloaded_dataset_path/VOCdevkit/VOC2012
```
After the script completes, pseudo labels are generated in the following directory and their quality is evaluated in mIoU. If you want to train DeepLab, add `--infer_list voc12/train_aug.txt` to the above script. The former and the latter respectively generate 1464 and 10582 pseudo-segmentation masks in `.png` format in the `.\result\seg_sem`.
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
2. Move `.\result\sem_seg` to `$downloaded_dataset_path/VOCdevkit/VOC2012`, and rename it to `pseudo_seg_labels`. You can actually move the folder, or make a soft link(recommanded).
```
ln -s .\result\sem_seg $downloaded_dataset_path/VOCdevkit/VOC2012\pseudo_seg_labels
```
The file structure of VOC2012 should look like this:
```
VOC2012
├─Annotations
├─ImageSets
│  ├─Action
│  ├─Layout
│  ├─Main
│  └─Segmentation
├─JPEGImages
├─SegmentationClass
├─SegmentationObject
└─pseudo_seg_labels
```
### Train DeepLab with the generated pseudo labels.
1. Download the [pretrained models](https://drive.google.com/file/d/1lwbyAo-XTKsmQX5ZtaZv-5DQONbhJh9n/view?usp=sharing) and put them into the `pretrained` folder.
2. Train DeepLabv2-resnet101 model.
```
python main.py train \
      --config-path configs/voc12_resnet_dplv2.yaml
```
3. Evaluate DeepLabv2-resnet101 model on the validation set.
```
python main.py test \
    --config-path configs/voc12_resnet_dplv2.yaml \
    --model-path data/models/voc12/voc12_resnet_v2/train_aug/checkpoint_final.pth
```
4. Re-evaluate with a CRF post-processing.
```
python main.py crf \
    --config-path configs/voc12_resnet_dplv2.yaml
```
## Performance
### Quality

<div align="center">
  <img src="fig\mask.png" width="800px">
  <p>Visualization of pseudo-segmentation masks on the PASCAL VOC 2012 training set.<br /> a) Input image; b) Ground truth; c) IRNet; d) TS-CAM; e) CRT</p>
</div>
<br>
<div align="center">
  <img src="fig\val.png" width="800px">
  <p>Visualization of pseudo-segmentation masks on the PASCAL VOC 2012 val set.<br /> a) Input image; b) Ground truth; c) CRT</p>
</div>

### Quantity
#### Pseudo segmentation mask
 Dataset | Seed | Mask | Weight
 ----- | ----- | ----- | -----
 PASCAL VOC | 57.7 | 71.8 | [Download](https://drive.google.com/file/d/1AItCgPldrAp929OLmfQkP9wwXC1oWeMz/view?usp=sharing)
#### WSSS results
 Dataset | Val | Test | Weight
 ----- | ----- | ----- | -----
 PASCAL VOC | 71.2 | 71.7 | [Download](https://drive.google.com/file/d/1sRhOUaPw5Rrx7IcmKMWf-enFroeF3Hsg/view?usp=sharing)
#### WSOL results
  Dataset | Top-1 | Top-5 | Gt-Known | Weight
 ----- | ----- | ----- | ----- | -----
  CUB-200-2011 | 72.9 | 86.4 | 90.1 | [Download](https://drive.google.com/file/d/1KCQC49zyaY2uD9n-CFVS88mUHVM1l5eB/view?usp=sharing)
## TODO
* link of paper
* picture of crt
* performance(quality)
* WSOL tutorial
* complete test(deeplab + tscam + model weight)
* code of deeplab, wsol

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

- [IRNet](https://github.com/jiwoon-ahn/irn), [TSCAM](https://github.com/vasgaowei/TS-CAM), [L2G](https://github.com/PengtaoJiang/L2G) and [Deeplab](https://github.com/kazuto1011/deeplab-pytorch).

