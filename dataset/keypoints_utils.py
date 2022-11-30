import sys
import os
sys.path.append(os.getcwd())
import math
import json
import math

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


################################################################################################################
# Single-Stage Multi-Person Pose Machines Utils
################################################################################################################
class SPMHeatmapGenerator:
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, (x, y) in enumerate(p):
                if x <= 0 and y <= 0:
                    continue
                
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

        return hms


class SPMMaskGenerator:
    def __init__(self, output_res, sigma=-1):
        self.output_res = output_res
        if sigma < 0:
            sigma = self.output_res/64
        self.size = int((6*sigma + 2) / 2)

    def __call__(self, joints):
        mask = np.zeros((len(joints), self.output_res, self.output_res), dtype=np.float32)
        for i, joint in enumerate(joints):
            for (x, y) in joint:
                if x <= 0 and y <= 0:
                    continue
                
                xmin = max(0, x - self.size)
                ymin = max(0, y - self.size)
                xmax = min(self.output_res, x + self.size + 1)
                ymax = min(self.output_res, y + self.size + 1)

                mask[i, ymin:ymax, xmin:xmax] = 1.

        return mask


class SPMDisplacementGenerator:
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints
        x_idx = torch.arange(0, output_res).repeat(output_res, 1)
        y_idx = x_idx.transpose(0, 1)
        self.x_idx = x_idx.numpy()
        self.y_idx = y_idx.numpy()
        self.z = math.sqrt(output_res**2 + output_res**2)

    def __call__(self, joints, masks):
        disp = np.zeros((self.num_joints*2, self.output_res, self.output_res), dtype=np.float32)
        for i, joint in enumerate(joints):
            mask = masks[i]
            for j, (x, y) in enumerate(joint):
                if x <= 0 and y <= 0:
                    continue
                
                disp[(j*2)] += mask * (x - self.x_idx) / self.z
                disp[(j*2) + 1] += mask * (y - self.y_idx) / self.z
                
        return disp


def nms_spm(heatmaps, conf_threshold=0.8, dist_threshold=7.):
    """NMS heatmaps of the SPM model
    
    Convert heatmaps to root joints info

    Arguments:
        heatmaps (Tensor): heatmaps of the SPM model with shape  '(1, output_size, output_size)'
        conf_threshold (float): confidence threshold to remove heatmap
        dist_threshold (float): distance threshold to remove heatmap

    Returns:
        Tensor: root joints '[num_root_joints, 2]', specified as [x, y]
    """

    yy, xx = torch.where(heatmaps[0] > conf_threshold)

    heatmaps_confidence = heatmaps[0][yy, xx]
    argsort_index = torch.argsort(-heatmaps_confidence)
    sorted_heatmaps_confidence = heatmaps_confidence[argsort_index]
    sorted_yy = yy[argsort_index]
    sorted_xx = xx[argsort_index]

    if sorted_heatmaps_confidence.size(0) == 0:
        return sorted_heatmaps_confidence

    # heatmaps nms
    root_joints_confidence = [] 
    root_joints = []
    while True:
        chosen_confidence = sorted_heatmaps_confidence[0]
        chosen_y = sorted_yy[0]
        chosen_x = sorted_xx[0]

        root_joints_confidence.append(chosen_confidence)
        root_joints.append(torch.stack([chosen_x, chosen_y]))

        tmp_confidences = []
        tmp_yy = []
        tmp_xx = []

        for idx in range(1, len(sorted_heatmaps_confidence)):
            tmp_confidence = sorted_heatmaps_confidence[idx]
            tmp_y = sorted_yy[idx]
            tmp_x = sorted_xx[idx]
            
            # calculating distance
            d = math.sqrt((tmp_x - chosen_x)**2 + (tmp_y - chosen_y)**2)

            if d > dist_threshold:
                tmp_confidences.append(tmp_confidence)
                tmp_yy.append(tmp_y)
                tmp_xx.append(tmp_x)
            
        if tmp_confidences:
            sorted_heatmaps_confidence = tmp_confidences
            sorted_yy = tmp_yy
            sorted_xx = tmp_xx
        else:
            break
    
    # print(f'Root Joints Confidence: {root_joints_confidence}')
    
    return torch.stack(root_joints)


def get_spm_keypoints(root_joints, displacements, dist_threshold):
    """Get Body Joint Keypoints

    Arguments:
        root_joints (Tensor): root joints '[num_root_joints, 2]', specified as [x, y]
        displacements (Tensor): [(2*num_keypoints), output_size, output_size]
        dist_threshold (float): distance threshold to remove Joint

    Returns:
        Tensor: keypoints joint '[num_root_joints, num_keypoints, 2]', specified as [x, y]
    """
    num_keypoints, output_size, _ = displacements.size()
    num_keypoints = int(num_keypoints / 2)
    z = math.sqrt(output_size**2 + output_size**2)
    device = displacements.device
    
    if root_joints.size(0) == 0:
        return root_joints

    keypoints_joint = []
    for root_joint in root_joints:
        x, y = root_joint
        tmp_keypoints = []
        for i in range(num_keypoints):
            keypoints_x = displacements[(2*i)][y, x] * z + x
            keypoints_y = displacements[(2*i + 1)][y, x] * z + y
            
            # calculating distance
            d = math.sqrt((x - keypoints_x)**2 + (y - keypoints_y)**2)
            
            if d < dist_threshold:
                tmp_keypoints.append(torch.tensor([0, 0], device=device))
            else:
                # keypoints_x = keypoints_x * output_size + x
                # keypoints_y = keypoints_y * output_size + y
                tmp_keypoints.append(torch.stack([keypoints_x, keypoints_y]))
        keypoints_joint.append(torch.stack(tmp_keypoints))
    return torch.stack(keypoints_joint)


class DecodeSPM(nn.Module):
    '''Decode SPM predictions to center(root joints) & keypoints joint
    
    Arguments:
        input_size (Int): Image input size 
        sigma (Int): 2D Gaussian Filter, size = 6*sigma + 3
        conf_threshold (Float): root joint confidence threshold value
        pred (Bool): True - for Predictions, False - for Targets
        x (Tensor): Predictions: [batch, 1 + (2*num_keypoints), output_size, output_size] or 
                    Targets: [batch, 1 + (2*num_keypoints), output_size, output_size]

    Returns:
        root_joints (Tensor): root joints '[num_root_joints, 2]', specified as [x, y], scaled input size
        keypoints_joint (Tensor): keypoints joint '[num_root_joints, num_keypoints, 2]', specified as [x, y], scaled input size
    '''
    def __init__(self, input_size, sigma, conf_threshold, pred=True):
        super().__init__()
        self.input_size = input_size
        self.dist_threshold = (6*sigma + 2) / 2
        self.conf_threshold = conf_threshold
        self.pred = pred

    def forward(self, x):
        assert x.size(0) == 1

        output_size = x.size(-1)

        if self.pred:
            heatmaps = torch.sigmoid(x[0, 0:1, :, :])# [1, output_size, output_size]
            displacements = torch.tanh(x[0, 1:, :, :]) # [(2*num_keypoints), output_size, output_size]
            # displacements = torch.zeros((32, 128, 128)) # [(2*num_keypoints), output_size, output_size]
        else:
            heatmaps = x[0, 0:1, :, :] # [1, output_size, output_size]
            displacements = x[0, 1:, :, :] # [(2*num_keypoints), output_size, output_size]
            
            # min_value = torch.min(displacements)
            # max_value = torch.max(displacements)
            # print(f'min: {min_value}, max: {max_value}')

        root_joints = nms_spm(heatmaps, self.conf_threshold, self.dist_threshold)

        keypoints_joint = get_spm_keypoints(root_joints, displacements, 2)

        # convert joints output_size scale to input_size scale
        root_joints = root_joints * self.input_size / output_size
        keypoints_joint = keypoints_joint * self.input_size / output_size

        return root_joints, keypoints_joint

def get_tagged_img_spm(img, root_joints, keypoints_joint):
    '''Return Tagged Image
    
    Arguments:
        img (Numpy): Image Array of Numpy or OpenCV 
        root_joints (Tensor): root joints '[num_root_joints, 2]', specified as [x, y], scaled input size
        keypoints_joint (Tensor): keypoints joint '[num_root_joints, num_keypoints, 2]', specified as [x, y], scaled input size
    
    Returns:
        img (Numpy): Tagged Image Array of Numpy or OpenCV 
    '''
    tagged_img = img.copy()
    
    # Draw keypoints joint
    for joints in keypoints_joint:
        for x, y in joints:
            if x <= 0. and y <= 0.:
                continue
            x, y = int(x), int(y)
            cv2.circle(tagged_img, (x, y), 3, (255, 0, 0), -1)

    # Draw Root joints
    for x, y in root_joints:
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 3, (0, 0, 255), -1)


    return tagged_img


################################################################################################################
# Passenger Interaction System Pose Estimation Utils
################################################################################################################
class SBPmAPPIS(SBPmAPCOCO):
    def __init__(self, json_path, input_size, conf_threshold):
        super().__init__(json_path, input_size, conf_threshold)

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
            tmp_joints.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            self.result_list.append({
                "image_id": int(img_ids[idx]),
                "category_id": int(cat_ids[idx]),
                "keypoints": tmp_joints,
                "score": float(sum(tmp_confs) / joints.size(0))
            })

def get_pis_tagged_img_sbp(img, joints):
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
        [5, 6, 6] 
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
        cv2.line(tagged_img, (x1, y1),  (x2, y2), limb_colors[limb[-1]], 4)
        
    # Draw keypoints joints   
    for (x, y, conf) in joints:
        if conf < 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 4, (0, 0, 255), -1)    
    
    return tagged_img

class HandleGrip:
    '''PIS Handle Grip Utils
    
    Arguments:
        handle_roi (Tuple): ((x1, y1), (x2, y2)), Handle roi for handle grip. 2 points on Image.
    '''
    def __init__(self, handle_roi):
        self.handle_roi = handle_roi
    
    def get_handle_grip_result(self, point):
        '''
        Arguments:
            point (Tensor): joint point, (x, y)
        Returns:
            result (Bool): True - Handle Grip, False - No Handle Grip
        '''
        gradient = (self.handle_roi[0][1] - self.handle_roi[1][1]) / (self.handle_roi[0][0] - self.handle_roi[1][0])
        y_intercept = self.handle_roi[0][1] - (gradient * self.handle_roi[0][0])
        
        intersection_x = int((point[1] - y_intercept)/gradient)
        
        return point[0] > intersection_x

class FallingDown:
    '''PIS Falling Down Utils
    
    Arguments:
        neg_max (int): Max Negative Gradient Value
        pos_min (int): Min Positive Gradient Value
    '''
    def __init__(self, neg_max, pos_min):
        self.neg_max = neg_max
        self.pos_min = pos_min
        
    def get_falling_down_result(self, point1, point2):
        '''
        Arguments:
            point1 (Tensor): joint point, (x, y)
            point2 (Tensor): joint point, (x, y)
        Returns:
            result (Bool): True - Normal, False - Falling Down
        '''
        gradient = (point1[1] - point2[1]) / (point1[0] - point2[0] + 1e-6)
        
        return gradient < self.neg_max or self.pos_min < gradient
    