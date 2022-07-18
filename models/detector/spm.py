import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary
import timm

from utils.module_select import get_model


class SPM(nn.Module):
    def __init__(self, backbone, num_keypoints):
        super().__init__()

        self.backbone = backbone
        self.num_keypoints = num_keypoints


    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        b1 = self.backbone.layer1(x)
        b2 = self.backbone.layer2(b1)
        b3 = self.backbone.layer3(b2)
        b4 = self.backbone.layer4(b3)
        b5 = self.backbone.layer5(b4)

        return b5


if __name__ == '__main__':
    input_size = 512

    backbone = get_model('darknet19')()

    model = SPM(
        backbone=backbone,
        num_keypoints=16
    )

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')
