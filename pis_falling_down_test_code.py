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
from dataset.keypoints_utils import DecodeSBP, get_tagged_img_sbp
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
    img_h, img_w, _ = representative_image.shape

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    no_detect_list = []
    fall_gradient = []
    normal_gradient = []

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
        nose = pred_joints[0].cpu().numpy()
        left_shoulder = pred_joints[5].cpu().numpy()
        right_shoulder = pred_joints[6].cpu().numpy()
        if nose[-1] < 0 or left_shoulder[-1] < 0 or right_shoulder[-1] < 0:
            no_detect_list.append(f'{os.sep}'.join(org_img_path.split(os.sep)[:-2]))
            
            if org_img_path.split(os.sep)[-5] == 'normal':
                fn += 1
            else:
                fp += 1
            
            continue

        shoulder_center = ((left_shoulder + right_shoulder) / 2)[:2]
        gradient = (nose[1] - shoulder_center[1]) / (nose[0] - shoulder_center[0] + 1e-6)
        org = (800, 1000)
        y = 500
        x = int(np.clip((org[0]-(y/gradient)), 0, img_w-1))
        neg_max = -1
        pos_min = 8

        if org_img_path.split(os.sep)[-5] == 'normal':
            color = (0, 255, 0)
            
            # cv2.circle(representative_image, (nose[0], nose[1]), 2, color, -1)
            cv2.line(representative_image, org, (x, y), color, 2)
            normal_gradient.append(gradient)
            
            if gradient < neg_max or pos_min < gradient:
                tp += 1
            else:
                fn += 1
            
        else:
            color = (0, 0, 255)
            
            # cv2.circle(representative_image, (nose[0], nose[1]), 2, color, -1)
            cv2.line(representative_image, org, (x, y), color, 2)
            fall_gradient.append(gradient)
            
            if gradient < neg_max or pos_min < gradient:
                fp += 1
            else:
                tn += 1
    
    cv2.line(representative_image, org, (int(np.clip((org[0]-(y/neg_max)), 0, img_w-1)), y), (255, 0, 0), 2)
    cv2.line(representative_image, org, (int(np.clip((org[0]-(y/pos_min)), 0, img_w-1)), y), (255, 0, 0), 2)
    
    print(len(no_detect_list))
    for no_detect in list(dict.fromkeys(no_detect_list)):
        print(no_detect)
    print('')
    
    normal_gradient = np.array(normal_gradient)
    negative_gradient = normal_gradient[np.where(normal_gradient < 0)]
    positive_gradient = normal_gradient[np.where(normal_gradient > 0)]
    print(f'neg_max: {max(negative_gradient)}, pos_min: {min(positive_gradient)}')
    
    print(f'total: {tp+tn+fp+fn}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Accuracy: {((tp+tn)/(tp+tn+fp+fn)*100):.2f}%')
    
    cv2.imshow('Image', representative_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
