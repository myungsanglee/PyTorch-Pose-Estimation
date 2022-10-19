import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
from torchinfo import summary
from models.backbone.darknet import darknet19


class SPM(nn.Module):
    def __init__(self, backbone_features_module, num_keypoints):
        super().__init__()
        
        self.backbone_features_module = backbone_features_module
        self.num_keypoints = num_keypoints
        
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.spm_head = nn.Sequential(
            nn.Conv2d(512, 1 + 2*self.num_keypoints, 1, 1, bias=False)
        )
        
    def forward(self, x):
        # backbone forward
        x = self.backbone_features_module(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        
        x = self.spm_head(x)
        
        return x


if __name__ == '__main__':
    input_size = 512
    output_size = 256
    num_keypoints = 17
    
    tmp_input = torch.randn((1, 3, input_size, input_size))
    
    backbone_features_module = darknet19(features_only=True)
    model = SPM(backbone_features_module, num_keypoints)
    
    summary(model, input_size=(1, 3, input_size, input_size), device='cpu')

    a = model(tmp_input)    
    print(f'{a.size()}')
    