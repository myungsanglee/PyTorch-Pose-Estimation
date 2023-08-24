import math
import os
import json

import torch
from torch import nn
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        Tensor: root joints '[num_root_joints, 3]', specified as [x, y, conf]
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
    
    root_joints_confidence = torch.stack(root_joints_confidence)
    root_joints = torch.stack(root_joints)
    
    return torch.concat([root_joints, root_joints_confidence[..., None]], dim=-1)


def get_spm_keypoints(root_joints, displacements, dist_threshold):
    """Get Body Joint Keypoints

    Arguments:
        root_joints (Tensor): root joints '[num_root_joints, 3]', specified as [x, y, conf]
        displacements (Tensor): [(2*num_keypoints), output_size, output_size]
        dist_threshold (float): distance threshold to remove Joint

    Returns:
        Tensor: keypoints joint '[num_root_joints, num_keypoints, 3]', specified as [x, y, conf]
    """
    num_keypoints, output_size, _ = displacements.size()
    num_keypoints = int(num_keypoints / 2)
    z = math.sqrt(output_size**2 + output_size**2)
    device = displacements.device
    
    if root_joints.size(0) == 0:
        return root_joints

    keypoints_joint = []
    for root_joint in root_joints:
        x, y, conf = root_joint
        tmp_keypoints = []
        for i in range(num_keypoints):
            keypoints_x = displacements[(2*i)][y.long(), x.long()] * z + x
            keypoints_y = displacements[(2*i + 1)][y.long(), x.long()] * z + y
            
            # calculating distance
            d = math.sqrt((x - keypoints_x)**2 + (y - keypoints_y)**2)
            
            if d < dist_threshold:
                tmp_keypoints.append(torch.tensor([0., 0., 0.], device=device))
            else:
                tmp_keypoints.append(torch.stack([keypoints_x, keypoints_y, conf]))
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
        root_joints (Tensor): root joints '[num_root_joints, 3]', specified as [x, y, conf], scaled input size
        keypoints_joint (Tensor): keypoints joint '[num_root_joints, num_keypoints, 3]', specified as [x, y, conf], scaled input size
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

        keypoints_joint = get_spm_keypoints(root_joints, displacements, self.dist_threshold)

        # convert joints output_size scale to input_size scale
        root_joints[..., :2] = root_joints[..., :2] * self.input_size / output_size
        keypoints_joint[..., :2] = keypoints_joint[..., :2] * self.input_size / output_size

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


class SPMmAPCOCO:
    def __init__(self, json_path, input_size, sigma, conf_threshold):
        self.coco = COCO(json_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.decoder = DecodeSPM(input_size, sigma, conf_threshold, True)
        self.result_list = []

    def reset_states(self):
        self.result_list = []

    def update_state(self, target, y_pred):
        batch_size = y_pred.size(0)
        image_sizes = target['image_size']
        img_ids = target['image_id']
        cat_ids = target['category_id']
        
        for idx in range(batch_size):
            _, keypoints_joint = self.decoder(y_pred[idx:idx+1]) # [num_root_joints, num_keypoints, 3]
            
            # convert joints input_size scale to original image scale
            keypoints_joint[..., :1] *= (image_sizes[0][idx] / self.input_size)
            keypoints_joint[..., 1:2] *= (image_sizes[1][idx] / self.input_size)

            for joints in keypoints_joint:
                tmp_joints = []
                tmp_confs = []
                for x, y, conf in joints:
                    if x == 0. and y == 0.:
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
        if not self.result_list:
            return 0
        
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