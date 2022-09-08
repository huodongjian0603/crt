# crt
The PyTorch Code for our paper "Dual Progressive Transformations for Weakly Supervised Semantic Segmentation"[pdf]
## Abstract
Weakly supervised semantic segmentation (WSSS), which aims to mine the object regions by merely using class-level labels, is a challenging task in computer vision. The current state-of-the-art CNN-based methods usually adopt Class-Activation-Maps (CAMs) to highlight the potential areas of the object, however, they may suffer from the part-activated issues. To this end, we try an early attempt to explore the global feature attention mechanism of vision transformer in WSSS task. However, since the transformer lacks the inductive bias as in CNN models, it can not boost the performance directly and may yield the over-activated problems. To tackle these drawbacks, we propose a Convolutional Neural Networks Refined Transformer (CRT) to mine a globally complete and locally accurate class activation maps in this paper. To validate the effectiveness of our proposed method, extensive experiments are conducted on PASCAL VOC 2012 and CUB-200-2011 datasets. Experimental evaluations show that our proposed CRT achieves the new state-of-the-art performance on both the weakly supervised semantic segmentation task the weakly supervised object localization task, which outperform others by a large margin.
## Requirement
* Python 3.7
* PyTorch 1.1.0+
* NVIDIA GeForce RTX 2080Ti x2
## Usage
1. Download the repository.
```
hello world
```
