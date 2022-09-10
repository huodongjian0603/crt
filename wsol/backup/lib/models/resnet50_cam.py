import torch.nn as nn
import torch.nn.functional as F
# from misc import torchutils
from models import resnet50
import math
import random
import numpy as np
import torch

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 200, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x, return_cam=False):
        x = self.stage1(x)
        x = self.stage2(x).detach()
        x = self.stage3(x)
        x = self.stage4(x)

        if return_cam:
            cam = self.classifier(x)
            x = gap2d(x, keepdims=True)
            x = self.classifier(x)
            x = x.view(-1, 200)            
            return x, cam
        else:
            cam = self.classifier(x)
            x = gap2d(x, keepdims=True)
            x = self.classifier(x)
            x = x.view(-1, 200)            
            return x, cam

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False


    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))