import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn


class SPMLoss(nn.Module):
    """SPM Loss Function
    """
    
    def __init__(self):
        super().__init__()
        # weight factor to balance two kinds of losses
        self.lambda_root = 100
        self.lambda_disp = 1
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.sl1_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, 1 + (2*num_keypoints), output_size, output_size]
            target (tensor): [batch, 1 + (2*num_keypoints), output_size, output_size]

        Returns:
            loss (float): total loss values
        """
        batch_size = input.size(0)
        # [batch, 1 + (2*num_keypoints), output_size, output_size] to [batch, output_size, output_size, 1 + (2*num_keypoints)]
        prediction = input.permute(0, 2, 3, 1).contiguous()

        pred_root_joints = torch.sigmoid(prediction[..., 0]) # [batch, output_size, output_size]
        pred_displacements = torch.tanh(prediction[..., 1:]) # [batch, output_size, output_size, (2*num_keypoints)]
        
        mask, n_mask, true_root_joints, true_displacements = self.encode_target(target)
        if prediction.is_cuda:
            mask = mask.cuda()
            n_mask = n_mask.cuda()
            true_root_joints = true_root_joints.cuda()
            true_displacements = true_displacements.cuda()

        # ======================== #
        #   FOR Root Joints Loss   #
        # ======================== #
        # loss_root_joints = self.lambda_root * self.mse_loss(pred_root_joints, true_root_joints)
        loss_root_joints = self.lambda_root * self.mse_loss(pred_root_joints * mask[..., 0], true_root_joints)
        loss_no_root_joints = self.mse_loss(pred_root_joints * n_mask[..., 0], true_root_joints * n_mask[..., 0])

        # ===================================== #
        #   FOR Body Joint Displacement LOSS    #
        # ===================================== #
        # loss_displacements = 100 * self.sl1_loss(pred_displacements, true_displacements)
        loss_displacements = self.sl1_loss(pred_displacements * mask, true_displacements * mask)
        # loss_no_displacements = self.sl1_loss(pred_displacements * n_mask, true_displacements * n_mask)

        # loss = (loss_root_joints + loss_displacements + loss_no_root_joints) / batch_size
        # loss = (loss_root_joints + loss_no_root_joints + loss_displacements + loss_no_displacements) * batch_size
        loss = (loss_root_joints + loss_no_root_joints + loss_displacements) * batch_size
        # loss = (loss_root_joints + loss_no_root_joints) * batch_size
        # loss = (loss_root_joints) * batch_size

        # # ======================== #
        # #   FOR Root Joints Loss   #
        # # ======================== #
        # loss_root_joints = self.lambda_root * self.mse_loss(pred_root_joints * mask[..., 0], true_root_joints * mask[..., 0])
        # loss_no_root_joints = self.lambda_no_root * self.mse_loss(pred_root_joints * n_mask[..., 0], true_root_joints * n_mask[..., 0])

        # # ===================================== #
        # #   FOR Body Joint Displacement LOSS    #
        # # ===================================== #
        # loss_positive_displacements = self.lambda_disp_pos * self.mse_loss(pred_displacements * mask, true_displacements)

        # loss = (loss_root_joints + loss_no_root_joints + loss_positive_displacements) * batch_size

        return loss

    def encode_target(self, target):
        """SPM Loss Function

        Arguments:
            target (Tensor): [batch, 1 + (2*num_keypoints), output_size, output_size]
        
        Retruns:
            mask (Tensor): Root Joints Mask Tensor, [batch_size, output_size, output_size, 1]
            n_mask (Tensor): No Root Joints Mask Tensor, [batch_size, output_size, output_size, 1]
            true_root_joints (Tensor): Ground Truth Root Joints, [batch_size, output_size, output_size]
            true_displacements (Tensor): Ground Truth Joint Displacements, [batch_size, output_size, output_size, (2*num_keypoints)]
        """
        heatmaps = target[:, 0:1, :,  :]
        displacements = target[:, 1:, :, :]

        true_root_joints = heatmaps[:, 0, :, :] # [batch, output_size, output_size]

        mask = torch.where(true_root_joints > 0., 1., 0.).type(torch.float32) # [batch, output_size, output_size]
        n_mask = torch.where(true_root_joints > 0., 0., 1.).type(torch.float32) # [batch, output_size, output_size]
        mask = mask.unsqueeze(dim=-1) # [batch, output_size, output_size, 1]
        n_mask = n_mask.unsqueeze(dim=-1) # [batch, output_size, output_size, 1]
        
        true_displacements = displacements.permute(0, 2, 3, 1).contiguous() # [batch_size, output_size, output_size, (2*num_keypoints)]
                
        return mask, n_mask, true_root_joints, true_displacements    
