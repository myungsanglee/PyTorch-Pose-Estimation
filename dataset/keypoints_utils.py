import sys
import os
sys.path.append(os.getcwd())
import math

import torch
from torch import nn
import numpy as np
import cv2


# def collater(data):
#     """Data Loader에서 생성된 데이터를 동일한 shape으로 정렬해서 Batch로 전달

#     Arguments::
#         data (Dict): albumentation Transformed 객체
#         'image': list of Torch Tensor len == batch_size, item shape = ch, h, w
#         'bboxes': list of list([cx, cy, w, h, cid])

#     Returns:
#         Dict: 정렬된 batch data.
#         'img': image tensor, [batch_size, channel, height, width] shape
#         'annot': 동일 shape으로 정렬된 tensor, [batch_size, max_num_annots, 5(cx, cy, w, h, cid)] shape
#     """
#     imgs = [s['image'] for s in data]
#     bboxes = [torch.tensor(s['bboxes'])for s in data]
#     batch_size = len(imgs)

#     max_num_annots = max(annots.shape[0] for annots in bboxes)

#     if max_num_annots > 0:
#         padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1
#         for idx, annot in enumerate(bboxes):
#             if annot.shape[0] > 0:
#                 padded_annots[idx, :annot.shape[0], :] = annot
                
#     else:
#         padded_annots = torch.ones((batch_size, 1, 5)) * -1

#     return {'img': torch.stack(imgs), 'annot': padded_annots}


'''
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
참고
'''
class HeatmapGenerator:
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
            for idx, pt in enumerate(p):
                # if pt[2] > 0:
                x, y = pt[0], pt[1]
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

        return hms


class MaskGenerator:
    def __init__(self, output_res, sigma=-1):
        self.output_res = output_res
        if sigma < 0:
            sigma = self.output_res/64
        self.size = int((6*sigma + 2) / 2)

    def __call__(self, joints):
        mask = np.zeros((len(joints), self.output_res, self.output_res), dtype=np.float32)

        for i, joint in enumerate(joints):
            for j, (x, y) in enumerate(joint):
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue
                
                xmin = max(0, x - self.size)
                ymin = max(0, y - self.size)
                xmax = min(self.output_res, x + self.size + 1)
                ymax = min(self.output_res, y + self.size + 1)

                mask[i, ymin:ymax, xmin:xmax] = 1.

        return mask


class DisplacementGenerator:
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints
        x_idx = torch.arange(0, output_res).repeat(output_res, 1)
        y_idx = x_idx.transpose(0, 1)
        self.x_idx = x_idx.numpy()
        self.y_idx = y_idx.numpy()

    def __call__(self, joints, masks):
        disp = np.zeros((self.num_joints*2, self.output_res, self.output_res), dtype=np.float32)
        for i, joint in enumerate(joints):
            mask = masks[i]
            for j, (x, y) in enumerate(joint):
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue
                
                disp[(j*2)] += mask * (x - self.x_idx) / self.output_res
                disp[(j*2) + 1] += mask * (y - self.y_idx) / self.output_res
                
        return disp


class PoseHeatmapGenerator:
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


def nms_heatmaps(heatmaps, conf_threshold=0.8, dist_threshold=7.):
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


def get_keypoints(root_joints, displacements):
    """Get Body Joint Keypoints

    Arguments:
        root_joints (Tensor): root joints '[num_root_joints, 2]', specified as [x, y]
        displacements (Tensor): [(2*num_keypoints), output_size, output_size]

    Returns:
        Tensor: keypoints joint '[num_root_joints, num_keypoints, 2]', specified as [x, y]
    """
    num_keypoints, output_size, _ = displacements.size()
    num_keypoints = int(num_keypoints / 2)
    
    if root_joints.size(0) == 0:
        return root_joints

    keypoints_joint = []
    for root_joint in root_joints:
        x, y = root_joint
        tmp_keypoints = []
        for i in range(num_keypoints):
            keypoints_x = displacements[(2*i)][y, x] * output_size + x
            keypoints_y = displacements[(2*i + 1)][y, x] * output_size + y
            tmp_keypoints.append(torch.stack([keypoints_x, keypoints_y]))
        keypoints_joint.append(torch.stack(tmp_keypoints))
    return torch.stack(keypoints_joint)


class DecodePoseNet(nn.Module):
    '''Decode PoseNet predictions to center(root joints) & keypoints joint
    
    Arguments:
        input_size (Int): Image input size 
        sigma (Int): 2D Gaussian Filter, size = 6*sigma + 3
        conf_threshold (Float): root joint confidence threshold value
        pred (Bool): True - for Predictions, False - for Targets
        x (Tensor): Predictions: [batch, nstack, 1 + (2*num_keypoints), output_size, output_size] or 
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
            x = torch.mean(x, dim=1) # [batch, nstack, 1 + (2*num_keypoints), output_size, output_size] to [batch, 1 + (2*num_keypoints), output_size, output_size]
            heatmaps = torch.sigmoid(x[0, 0:1, :, :])# [1, output_size, output_size]
            displacements = torch.tanh(x[0, 1:, :, :]) # [(2*num_keypoints), output_size, output_size]
        else:
            heatmaps = x[0, 0:1, :, :] # [1, output_size, output_size]
            displacements = x[0, 1:, :, :] # [(2*num_keypoints), output_size, output_size]

        root_joints = nms_heatmaps(heatmaps, self.conf_threshold, self.dist_threshold)

        keypoints_joint = get_keypoints(root_joints, displacements)

        # convert joints output_size scale to input_size scale
        root_joints = root_joints * self.input_size / output_size
        keypoints_joint = keypoints_joint * self.input_size / output_size

        return root_joints, keypoints_joint


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
            # displacements = torch.tanh(x[0, 1:, :, :]) # [(2*num_keypoints), output_size, output_size]
            displacements = torch.zeros((32, 128, 128)) # [(2*num_keypoints), output_size, output_size]
        else:
            heatmaps = x[0, 0:1, :, :] # [1, output_size, output_size]
            displacements = x[0, 1:, :, :] # [(2*num_keypoints), output_size, output_size]
            
            # min_value = torch.min(displacements)
            # max_value = torch.max(displacements)
            # print(f'min: {min_value}, max: {max_value}')

        root_joints = nms_heatmaps(heatmaps, self.conf_threshold, self.dist_threshold)

        keypoints_joint = get_keypoints(root_joints, displacements)

        # convert joints output_size scale to input_size scale
        root_joints = root_joints * self.input_size / output_size
        keypoints_joint = keypoints_joint * self.input_size / output_size

        return root_joints, keypoints_joint


def nms_sbp(heatmaps, conf_threshold=0.8):
    """NMS heatmaps of the Simple Baseline Pose-Estimation model
    
    Convert heatmaps to joints info

    Arguments:
        heatmaps (Tensor): heatmaps of the SBP model with shape  '(num_keypoints, output_size, output_size)'
        conf_threshold (float): confidence threshold to remove heatmap

    Returns:
        Tensor: joints '[num_keypoints, 2]', specified as [x, y]
    """
    num_keypoints = heatmaps.size(0)
    joints = torch.zeros((num_keypoints, 2)) - 1
    
    for idx in torch.arange(num_keypoints):
        heatmap = heatmaps[idx]
        yy, xx = torch.where(heatmap > conf_threshold)
        if yy.size(0) == 0:
            continue
        
        heatmap_confidence = heatmap[yy, xx]
        argmax_index = torch.argmax(heatmap_confidence)

        joints[idx] = torch.tensor([xx[argmax_index], yy[argmax_index]])

    return joints


class DecodeSBP(nn.Module):
    '''Decode SBP predictions to joints
    
    Arguments:
        input_size (Int): Image input size 
        conf_threshold (Float): joint confidence threshold value
        pred (Bool): True - for Predictions, False - for Targets
        x (Tensor): [batch, num_keypoints, output_size, output_size]

    Returns:
        joints (Tensor): heatmap joints '[num_keypoints, 2]', specified as [x, y], scaled input size
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

        joints = nms_sbp(heatmaps[0], self.conf_threshold) # [num_keypoints, 2]

        # convert joints output_size scale to input_size scale
        joints = joints * self.input_size / output_size

        return joints
    

def get_tagged_img(img, root_joints, keypoints_joint):
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
            x, y = int(x), int(y)
            cv2.circle(tagged_img, (x, y), 3, (255, 0, 0), -1)

    # Draw Root joints
    for x, y in root_joints:
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 3, (0, 0, 255), -1)


    return tagged_img


def get_tagged_img_sbp(img, joints):
    '''Return Tagged Image
    
    Arguments:
        img (Numpy): Image Array of Numpy or OpenCV 
        joints (Tensor): joints '[num_keypoints, 2]', specified as [x, y], scaled input size
    
    Returns:
        img (Numpy): Tagged Image Array of Numpy or OpenCV 
    '''
    tagged_img = img.copy()
    h, w, _ = tagged_img.shape
    
    class_colors = [
        (0, 0, 255), # r_ankle
        (0, 255, 0), # r_knee
        (255, 0, 0), # r_hip
        (255, 255, 0), # l_hip
        (255, 0, 255), # l_knee
        (0, 255, 255), # l_ankle
        (255, 0, 153), # pelvis
        (102, 102, 0), # thorax
        (0, 0, 204), # upper_neck
        (0, 0, 0), # head_top
        (0, 0, 255), # r_wrist
        (0, 255, 0), # r_elbow
        (255, 0, 0), # r_shoulder
        (255, 255, 0), # l_shoulder
        (255, 0, 255), # l_elbow
        (0, 255, 255) # l_wrist
    ]
    
    # Draw keypoints joint
    for idx, (x, y) in enumerate(joints):
        if x < 0 or y < 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 3, (0, 255, 0), -1)
        
        margin = 10
        org_x = np.clip(x, margin, w - margin)
        org_y = np.clip(y, margin, h - margin)
        cv2.putText(tagged_img, f'{idx}', (org_x, org_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        
    return tagged_img