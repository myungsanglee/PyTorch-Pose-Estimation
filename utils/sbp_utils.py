import os
import json

import torch
from torch import nn
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


'''
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
참고
'''

################################################################################################################
# Simple Baselines Pose Estimation Utils
################################################################################################################
class SBPHeatmapGenerator:
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res_h, self.output_res_w = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res_h/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res_h, self.output_res_w), dtype=np.float32)
        sigma = self.sigma
        for idx, (x, y) in enumerate(joints):
            if x < 0 or y < 0:
                continue
            
            x, y = np.clip(int(x), 0, self.output_res_w - 1), np.clip(int(y), 0, self.output_res_h - 1)
            
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res_w) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res_h) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res_w)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res_h)
            
            hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
            
        return hms


def nms_sbp(heatmaps, conf_threshold=0.8):
    """NMS heatmaps of the Simple Baseline Pose-Estimation model
    
    Convert heatmaps to joints info

    Arguments:
        heatmaps (Tensor): heatmaps of the SBP model with shape  '(num_keypoints, output_size, output_size)'
        conf_threshold (float): confidence threshold to remove heatmap

    Returns:
        Tensor: joints '[num_keypoints, 3]', specified as [x, y, confidence]
    """
    num_keypoints = heatmaps.size(0)
    joints = torch.zeros((num_keypoints, 3), device=heatmaps.device) - 1
    
    for idx in torch.arange(num_keypoints):
        heatmap = heatmaps[idx]
        yy, xx = torch.where(heatmap > conf_threshold)
        if yy.size(0) == 0:
            continue
        
        heatmap_confidence = heatmap[yy, xx]
        argmax_index = torch.argmax(heatmap_confidence)

        joints[idx] = torch.tensor([xx[argmax_index], yy[argmax_index], heatmap_confidence[argmax_index]])

    return joints


class DecodeSBP(nn.Module):
    '''Decode SBP predictions to joints
    
    Arguments:
        input_size (Int): Image input size 
        conf_threshold (Float): joint confidence threshold value
        pred (Bool): True - for Predictions, False - for Targets
        x (Tensor): [batch, num_keypoints, output_size, output_size]

    Returns:
        joints (Tensor): heatmap joints '[num_keypoints, 3]', specified as [x, y, confidence], scaled input size
    '''
    def __init__(self, input_size, conf_threshold, pred=True):
        super().__init__()
        self.input_size = input_size[-1]
        self.conf_threshold = conf_threshold
        self.pred = pred
        
    def forward(self, x):
        assert x.size(0) == 1

        output_size = x.size(-1)

        if self.pred:
            heatmaps = torch.sigmoid(x) # [batch, num_keypoints, output_size, output_size]
        else:
            heatmaps = x

        joints = nms_sbp(heatmaps[0], self.conf_threshold) # [num_keypoints, 3]

        # convert joints output_size scale to input_size scale
        joints[..., :2] *= (self.input_size / output_size)

        return joints


class SBPmAPCOCO:
    def __init__(self, json_path, input_size, conf_threshold):
        self.coco = COCO(json_path)
        self.input_size = input_size
        self.decoder = DecodeSBP(input_size, conf_threshold, True)
        self.result_list = []

    def reset_states(self):
        self.result_list = []

    def update_state(self, target, y_pred):
        batch_size = y_pred.size(0)
        bbox = target['bbox']
        img_ids = target['image_id']
        cat_ids = target['category_id']
        
        for idx in range(batch_size):
            joints = self.decoder(y_pred[idx:idx+1]) # [num_keypoints, 3]
            
            # convert joints input_size scale to original image scale
            joints[..., :1] *= (bbox[idx][2] / self.input_size[1])
            joints[..., 1:2] *= (bbox[idx][3] / self.input_size[0])

            # convert joints to original image coordinate
            joints[..., :1] += bbox[idx][0]
            joints[..., 1:2] += bbox[idx][1]
            
            tmp_joints = []
            tmp_confs = []
            for (x, y, conf) in joints:
                if conf < 0:
                    tmp_joints.extend([0, 0, 0])
                    tmp_confs.append(0)
                    continue
                
                tmp_joints.extend([float(x), float(y), 1])
                tmp_confs.append(conf)
            
            self.result_list.append({
                "image_id": int(img_ids[idx]),
                "category_id": int(cat_ids[idx]),
                "keypoints": tmp_joints,
                "score": float(sum(tmp_confs) / joints.size(0))
            })

    def result(self):
        results_json_path = os.path.join(os.getcwd(), 'results.json')
        with open(results_json_path, "w") as f:
            json.dump(self.result_list, f, indent=4)

        img_ids = sorted(self.coco.getImgIds())
        cat_ids = sorted(self.coco.getCatIds())
        
        # load detection JSON file from the disk
        cocovalPrediction = self.coco.loadRes(results_json_path)
        # initialize the COCOeval object by passing the coco object with
        # ground truth annotations, coco object with detection results
        cocoEval = COCOeval(self.coco, cocovalPrediction, "keypoints")
        
        # run evaluation for each image, accumulates per image results
        # display the summary metrics of the evaluation
        cocoEval.params.imgIds  = img_ids
        cocoEval.params.catIds  = cat_ids
    
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval.stats[1]

def get_coco_tagged_img_sbp(img, joints):
    '''Return Tagged Image
    
    Arguments:
        img (Numpy): Image Array of Numpy
        joints (Tensor): joints '[num_keypoints, 3]', specified as [x, y, conf]
    
    Returns:
        img (Numpy): Tagged Image Array of Numpy
    '''
    tagged_img = img.copy()
    h, w, _ = tagged_img.shape
    
    limb_colors = [
        (0, 102, 102), # right face
        (102, 0, 102), # left face
        (0, 204, 0), # right arm
        (204, 0, 0), # left arm
        (0, 102, 0), # right leg
        (102, 0, 0), # left leg
        (0, 0, 0) # others
    ]

    # [joint_idx, joint_idx, limb_color_idx]
    joint_limbs = [
        [0, 1, 1], 
        [0, 2, 0], 
        [1, 3, 1], 
        [2, 4, 0], 
        [5, 7, 3], 
        [6, 8, 2], 
        [7, 9, 3], 
        [8, 10, 2], 
        [11, 13, 5], 
        [12, 14, 4], 
        [13, 15, 5], 
        [14, 16, 4], 
        [5, 6, 6], 
        [5, 11, 6], 
        [6, 12, 6], 
        [11, 12, 6], 
    ]
    
    # Draw keypoints limbs
    for limb in joint_limbs:
        tmp_joints = joints[limb[:2]]
        joint1 = tmp_joints[0]
        joint2 = tmp_joints[1]
        if joint1[-1] < 0 or joint2[-1] < 0:
            continue
        x1, y1 = int(joint1[0]), int(joint1[1])
        x2, y2 = int(joint2[0]), int(joint2[1])
        cv2.line(tagged_img, (x1, y1),  (x2, y2), limb_colors[limb[-1]], 2)
        
    # Draw keypoints joints   
    for (x, y, conf) in joints:
        if conf < 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 2, (0, 0, 255), -1)    
    
    return tagged_img
