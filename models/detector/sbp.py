import os
import sys
sys.path.append(os.getcwd())

from torch import nn
from torchinfo import summary
from models.backbone.darknet import darknet19


class SBP(nn.Module):
    def __init__(self, backbone_module_list, num_keypoints):
        super().__init__()
        
        self.backbone_module_list = backbone_module_list
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
        
        self.sbp_head = nn.Sequential(
            nn.Conv2d(512, self.num_keypoints, 1, 1, bias=False)
        )
        
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        # backbone forward
        for idx, module in enumerate(self.backbone_module_list):
            x = module(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        
        x = self.dropout(x)
        
        x = self.sbp_head(x)

        return x


if __name__ == '__main__':
    input_size = [256, 192] # [height, width]
    output_size = [64, 48] # [height, width]

    backbone = darknet19(pretrained='')
    model = SBP(backbone.get_features_module_list(), 16)

    summary(model, (1, 3, input_size[0], input_size[1]), device='cpu')
    