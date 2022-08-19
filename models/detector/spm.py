import os
import sys
from turtle import forward
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary
from models.layers.blocks import Conv, Hourglass, Pool, Residual
from models.backbone.darknet import darknet19


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack

    def forward(self, x):
        ## our posenet
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)


class SPM(nn.Module):
    def __init__(self, backbone, num_keypoints):
        super().__init__()
        
        self.backbone = backbone
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
        
        # self.spm_head = nn.Sequential(
        #     nn.Conv2d(128, 1 + 2*self.num_keypoints, 1, 1, bias=False)
        #     # nn.Conv2d(128, 1, 1, 1, bias=False)
        # )
        
        self.root_head = nn.Sequential(
            nn.Conv2d(512, 1, 1, 1, bias=False)
        )
        
        self.disp_head = nn.Sequential(
            nn.Conv2d(512, 2*self.num_keypoints, 1, 1, bias=False)
        )
        
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        b1 = self.backbone.layer1(x)
        b2 = self.backbone.layer2(b1)
        b3 = self.backbone.layer3(b2)
        b4 = self.backbone.layer4(b3)
        b5 = self.backbone.layer5(b4)
        
        x = self.deconv_1(b5)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        
        x = self.dropout(x)
        
        # x = self.spm_head(x)
        
        root = self.root_head(x)

        return root
        
        disp = self.disp_head(x)
        
        x = torch.cat((root, disp), dim=1)

        return x


if __name__ == '__main__':
    input_size = 512
    output_size = 256

    tmp_input = torch.randn((1, 3, input_size, input_size))

    # model = PoseNet(nstack=8, inp_dim=256, oup_dim=33)
    
    backbone = darknet19()
    model = SPM(backbone, 16)

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')

    a = model(tmp_input)    
    print(f'{a.size()}')
