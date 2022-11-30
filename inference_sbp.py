import argparse
import time
import os

import torch
import numpy as np
import cv2

from utils.yaml_helper import get_configs
from module.sbp_detector import SBPDetector
from models.detector.sbp import SBP
from utils.module_select import get_model
from dataset.keypoints_utils import DecodeSBP, get_coco_tagged_img_sbp
from dataset.sbp_coco_dataset import SBPCOCODataModule


def inference(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = SBPCOCODataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        img_dir = cfg['img_dir'],
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
    
    model_module = SBPDetector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    pred_decoder = DecodeSBP(cfg['input_size'], cfg['conf_threshold'], True)
    true_decoder = DecodeSBP(cfg['input_size'], 0.99, False)

    # Inference
    for img, target in data_module.val_dataloader():
    # for img, target in data_module.train_dataloader():
        if torch.cuda.is_available:
            img = img.cuda()
        
        bbox = target['bbox'][0]
        org_img_path = target['image_path'][0]
        org_img = cv2.imread(org_img_path)
        
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

        # batch_x to img
        # if torch.cuda.is_available:
        #     img = img.cpu()[0].numpy()   
        # else:
        #     img = img[0].numpy()   
        # img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        pred_img = get_coco_tagged_img_sbp(org_img, pred_joints)
        true_img = get_coco_tagged_img_sbp(org_img, true_joints)

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
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
