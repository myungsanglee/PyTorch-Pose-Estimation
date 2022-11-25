import argparse
import time
import os

import torch
import numpy as np
import cv2

from utils.yaml_helper import get_configs
from module.sbp_pis_detector import SBPPISDetector
from models.detector.sbp import SBP
from utils.module_select import get_model
from dataset.keypoints_utils import DecodeSBP
from dataset.sbp_pis_dataset import SBPPISDataModule


def inference(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = SBPPISDataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        input_size = cfg['input_size'],
        output_size = cfg['output_size'],
        num_keypoints = cfg['num_keypoints'],
        sigma = cfg['sigma'],
        workers = cfg['workers'],
        batch_size = 1,
        class_labels=cfg['class_labels']
    )
    data_module.prepare_data()
    data_module.setup()
    
    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'],
        features_only=True
    )
    
    model = SBP(
        backbone_features_module=backbone_features_module, 
        num_keypoints=cfg['num_keypoints']
    )

    if torch.cuda.is_available:
        model = model.cuda()
    
    model_module = SBPPISDetector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    pred_decoder = DecodeSBP(cfg['input_size'], cfg['conf_threshold'], True)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1280, 960)
    representative_image = cv2.imread('/home/fssv2/myungsang/datasets/pis/representative_image.jpg')

    # set handle region
    handle_roi = ((1220, 1300), (1600, 1130))
    representative_image = cv2.line(representative_image, handle_roi[0], handle_roi[1], color=(255, 0, 0), thickness=2)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    right_flag = False

    # Inference
    for img, target in data_module.val_dataloader():
        if torch.cuda.is_available:
            img = img.cuda()
        
        bbox = target['bbox'][0]
        org_img_path = target['image_path'][0]
        
        with torch.no_grad():
            predictions = model_module(img)
        pred_joints = pred_decoder(predictions)

        # convert joints input_size scale to original image scale
        pred_joints[..., :1] *= (bbox[2] / cfg['input_size'][1])
        pred_joints[..., 1:2] *= (bbox[3] / cfg['input_size'][0])

        # convert joints to original image coordinate
        pred_joints[..., :1] += bbox[0]
        pred_joints[..., 1:2] += bbox[1]
        
        # Draw handle point
        right_hand = pred_joints[10]

        if org_img_path.split(os.sep)[-5] == 'grip':
            color = (0, 255, 0)
            
            if right_hand[-1] < 0:
                fn += 1
            else:
                cv2.circle(representative_image, (right_hand[0], right_hand[1]), 2, color, -1)
                right_flag = get_handle_grip_result(handle_roi, right_hand[:2])
                
                if right_flag:
                    tp += 1
                else:
                    fn += 1
        else:
            color = (0, 0, 255)
            
            if right_hand[-1] < 0:
                fp += 1
            else:
                cv2.circle(representative_image, (right_hand[0], right_hand[1]), 2, color, -1)
                right_flag = get_handle_grip_result(handle_roi, right_hand[:2])
                
                if not right_flag:
                    tn += 1
                else:
                    fp += 1

    print(f'total: {tp+tn+fp+fn}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Accuracy: {((tp+tn)/(tp+tn+fp+fn)*100):.2f}%')

    cv2.imshow('Image', representative_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def get_handle_grip_result(region, point):
    gradient = (region[0][1] - region[1][1]) / (region[0][0] - region[1][0])
    y_intercept = region[0][1] - (gradient * region[0][0])
    
    intersection_x = int((point[1] - y_intercept)/gradient)
    
    return point[0] > intersection_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
