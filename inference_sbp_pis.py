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
from utils.sbp_utils import DecodeSBP
from utils.sbp_pis_utils import get_pis_tagged_img_sbp, HandleGrip, FallingDown
from dataset.sbp_pis_dataset import SBPPISDataModule


def inference(cfg, ckpt, pis):
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
    true_decoder = DecodeSBP(cfg['input_size'], 0.99, False)

    cv2.namedWindow('pred', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('pred', 1280, 960)
    cv2.namedWindow('true', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('true', 1280, 960)
    
    # PIS
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 2
    if pis == 'handle_grip':
        handle_roi = ((1220, 1300), (1600, 1130))
        handle_cls = HandleGrip(handle_roi)
    elif pis == 'falling_down':
        neg_max = -1
        pos_min = 8
        falling_cls = FallingDown(neg_max, pos_min)

    # Inference
    for img, target in data_module.val_dataloader():
        if torch.cuda.is_available:
            img = img.cuda()
        
        bbox = target['bbox'][0]
        org_img_path = target['image_path'][0]
        org_img = cv2.imread(org_img_path)
        img_h, img_w, _ = org_img.shape
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(img)
        pred_joints = pred_decoder(predictions)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        
        true_joints = true_decoder(target['heatmaps'])

        # convert joints input_size scale to original image scale
        pred_joints[..., :1] *= (bbox[2] / cfg['input_size'][1])
        pred_joints[..., 1:2] *= (bbox[3] / cfg['input_size'][0])
        
        true_joints[..., :1] *= (bbox[2] / cfg['input_size'][1])
        true_joints[..., 1:2] *= (bbox[3] / cfg['input_size'][0])

        # convert joints to original image coordinate
        pred_joints[..., :1] += bbox[0]
        pred_joints[..., 1:2] += bbox[1]
        
        true_joints[..., :1] += bbox[0]
        true_joints[..., 1:2] += bbox[1]
        
        # pis
        if pis == 'handle_grip':
            org_img = cv2.line(org_img, handle_roi[0], handle_roi[1], color=(255, 0, 0), thickness=2)
            
            if pred_joints[10][-1] < 0:
                handle_result = False
            else:
                handle_result = handle_cls.get_handle_grip_result(pred_joints[10][:2])
            
            if handle_result:
                txt = 'Handle Grip'
                size, _ = cv2.getTextSize(txt, font, font_scale, thickness)
                org = (int((img_w - size[0])/2), 300)
                org_img = cv2.putText(org_img, txt, org, font, font_scale, (0, 255, 0), thickness)

            else:
                txt = 'No Handle Grip'
                size, _ = cv2.getTextSize(txt, font, font_scale, thickness)
                org = (int((img_w - size[0])/2), 300)
                org_img = cv2.putText(org_img, txt, org, font, font_scale, (0, 0, 255), thickness)

        elif pis == 'falling_down':
            nose = pred_joints[0]
            left_shoulder = pred_joints[5]
            right_shoulder = pred_joints[6]
            
            if nose[-1] < 0 or left_shoulder[-1] < 0 or right_shoulder[-1] < 0:
                falling_result = True
            else:
                shoulder_center = (left_shoulder + right_shoulder) / 2
                falling_result = falling_cls.get_falling_down_result(shoulder_center[:2], nose[:2])

            if falling_result:
                txt = 'Normal'
                size, _ = cv2.getTextSize(txt, font, font_scale, thickness)
                org = (int((img_w - size[0])/2), 300)
                org_img = cv2.putText(org_img, txt, org, font, font_scale, (0, 255, 0), thickness)

            else:
                txt = 'Falling Down'
                size, _ = cv2.getTextSize(txt, font, font_scale, thickness)
                org = (int((img_w - size[0])/2), 300)
                org_img = cv2.putText(org_img, txt, org, font, font_scale, (0, 0, 255), thickness)

        pred_img = get_pis_tagged_img_sbp(org_img, pred_joints)
        true_img = get_pis_tagged_img_sbp(org_img, true_joints)

        cv2.imshow('true', true_img)
        cv2.imshow('pred', pred_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    parser.add_argument('--pis', required=False, type=str, choices=['handle_grip', 'falling_down'], help='Show PIS Functions')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)
    
    inference(cfg, args.ckpt, args.pis)
