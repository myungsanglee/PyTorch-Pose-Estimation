import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn


class SBPLoss(nn.Module):
    """Simple Baseline Pose-Estimation Loss Function
    """
    
    def __init__(self):
        super().__init__()
        # weight factor to balance two kinds of losses
        self.lambda_positive = 100
        self.lambda_negative = 1
        
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, num_keypoints, output_size, output_size]
            target (tensor): [batch, num_keypoints, output_size, output_size]

        Returns:
            loss (float): total loss values
        """
        batch_size = input.size(0)
        # [batch, num_keypoints, output_size, output_size] to [batch, output_size, output_size, num_keypoints]
        prediction = input.permute(0, 2, 3, 1).contiguous()
        prediction = torch.sigmoid(prediction) 
                
        mask, n_mask, true_heatmaps = self.encode_target(target)
        if prediction.is_cuda:
            mask = mask.cuda()
            n_mask = n_mask.cuda()
            true_heatmaps = true_heatmaps.cuda()

        # ======================== #
        #   Joints Heatmap Loss   #
        # ======================== #
        loss_positive = self.lambda_positive * self.mse_loss(prediction * mask, true_heatmaps)
        loss_negative = self.lambda_negative * self.mse_loss(prediction * n_mask, true_heatmaps * n_mask)

        loss = (loss_positive + loss_negative) * batch_size
        
        # loss_joints = self.mse_loss(prediction, true_heatmaps)

        # loss = loss_joints * batch_size

        return loss

    def encode_target(self, target):
        """
        Arguments:
            target (Tensor): [batch, num_keypoints, output_size, output_size]
        
        Retruns:
            mask (Tensor): Heatmaps Mask Tensor, [batch_size, output_size, output_size, 1]
            n_mask (Tensor): No Heatmaps Mask Tensor, [batch_size, output_size, output_size, 1]
            true_heatmaps (Tensor): Ground Truth Heatmaps, [batch_size, output_size, output_size, num_keypoints]
        """
        true_heatmaps = target.permute(0, 2, 3, 1).contiguous() # [batch_size, output_size, output_size, num_keypoints]
        
        mask = torch.where(true_heatmaps > 0., 1., 0.).type(torch.float32) # [batch_size, output_size, output_size, num_keypoints]
        n_mask = torch.where(true_heatmaps > 0., 0., 1.).type(torch.float32) # [batch_size, output_size, output_size, num_keypoints]
                 
        return mask, n_mask, true_heatmaps  
    