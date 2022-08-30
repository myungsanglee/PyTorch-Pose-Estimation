import sys
import os
sys.path.append(os.getcwd())
from glob import glob
import json

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations as A

from dataset.keypoints_utils import PoseHeatmapGenerator, DecodeSBP
from utils.yaml_helper import get_configs


class SBPDataset(Dataset):
    def __init__(self, img_dir, file_path, transforms, heatmap_generator, ratio):
        super().__init__()

        self.img_dir = img_dir
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.ratio = ratio # output_size / input_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anno = self.data[index]
        
        # get image 
        img_filename = anno['image']
        img_file = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # get bounding box coordinates
        joints_vis = np.array(anno['joints_vis'])
        joints = np.array(anno['joints'])
        joints_vis, joints = self._check_joints(joints_vis, joints, (w, h)) 
        vis_idx = np.where(joints_vis > 0)[0]
        
        xmin = np.min(joints[vis_idx, :1])
        xmax = np.max(joints[vis_idx, :1])
        ymin = np.min(joints[vis_idx, 1:])
        ymax = np.max(joints[vis_idx, 1:])
        
        xmin = int(xmin - (xmax - xmin) * 0.15)
        xmax = int(xmax + (xmax - xmin) * 0.15)
        ymin = int(ymin - (ymax - ymin) * 0.05)
        ymax = int(ymax + (ymax - ymin) * 0.1)
        
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        
        # get cropped image
        cropped_img = img[ymin:ymax+1, xmin:xmax+1]
        
        # transform joints coordinates
        joints[vis_idx, :1] -= xmin
        joints[vis_idx, 1:] -= ymin
        # joints[np.where(joints_vis < 1)[0], :] = 0

        # # transform
        transformed = self.transforms(image=cropped_img, keypoints=joints)

        transformed_img = transformed['image']
        transformed_keypoints = np.array(transformed['keypoints'])

        # convert image [height, width, channel] to [channel, height, width]
        transformed_img = np.transpose(transformed_img, (2, 0, 1))

        # convert 'keypoints coordinates' input_size ratio to ouput_size ratio
        keypoints = transformed_keypoints * self.ratio
        keypoints[np.where(joints_vis < 1)[0], :] = -1

        # get heatmaps of root joints
        heatmaps = self.heatmap_generator(keypoints)

        return transformed_img, heatmaps
    
    def _check_joints(self, joints_vis, joints, img_size):
        w, h = img_size
        tmp_joints_vis = []
        tmp_joints = []
        for vis, joint in zip(joints_vis, joints):
            if vis:
                x, y = joint[0], joint[1]
                if x < 0 or y < 0:
                    # print(image)
                    tmp_joints.append([0, 0])
                    tmp_joints_vis.append(0)

                else:
                    tmp_joints.append([np.clip(int(x), 0, w - 1), np.clip(int(y), 0, h - 1)])
                    tmp_joints_vis.append(vis)
                
            else:
                tmp_joints.append([0, 0])
                tmp_joints_vis.append(0)

        return np.array(tmp_joints_vis), np.array(tmp_joints)


class SBPDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_path, 
        val_path,
        img_dir,
        input_size,
        output_size,
        num_keypoints,
        sigma,
        workers,
        batch_size
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.img_dir = img_dir
        self.workers = workers
        self.input_size = input_size
        self.output_size = output_size
        self.num_keypoints = num_keypoints
        self.batch_size = batch_size
        self.heatmap_generator = PoseHeatmapGenerator(
            output_size, self.num_keypoints, sigma
        )
        self.ratio = self.output_size[0] / self.input_size[0]
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(0, 1)
        ], keypoint_params=A.KeypointParams(format='xy'))
        
        valid_transform = A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(0, 1)
        ], keypoint_params=A.KeypointParams(format='xy'))
        
        self.train_dataset = SBPDataset(
            self.img_dir,
            self.train_path,
            train_transforms, 
            self.heatmap_generator,
            self.ratio
        )
        
        self.valid_dataset = SBPDataset(
            self.img_dir,
            self.val_path,
            valid_transform, 
            self.heatmap_generator,
            self.ratio
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
        )


if __name__ == '__main__':
    cfg = get_configs('./configs/pose_mpii.yaml')

    data_module = SBPDataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        img_dir = cfg['img_dir'],
        input_size = cfg['input_size'],
        output_size = cfg['output_size'],
        num_keypoints = cfg['num_keypoints'],
        sigma = cfg['sigma'],
        workers = cfg['workers'],
        batch_size = 1,
    )
    data_module.prepare_data()
    data_module.setup()

    sbp_decoder = DecodeSBP(cfg['input_size'], 0.99, False)

    for img, target in data_module.train_dataloader():
    # for img, target in data_module.val_dataloader():        
        # convert img to opencv numpy array
        img = img[0].permute((1, 2, 0)).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        heatmaps = target
        print(heatmaps.size())
        masks = torch.where(heatmaps > 0., 1., 0.).type(torch.float32)
        print(masks.size())
        
        joints = sbp_decoder(heatmaps)
        
        # Draw keypoints joint
        for (x, y) in joints:
            if x < 0 or y < 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        
        heatmaps = heatmaps[0].permute((1, 2, 0)).numpy()
        # heatmaps = np.sum(heatmaps, axis=-1)
        # heatmaps = cv2.resize(heatmaps, (192, 256))
        
        masks = masks[0].permute((1, 2, 0)).numpy()

        # heatmaps = cv2.resize(heatmaps, (416, 416))
        # masks = cv2.resize(masks, (416, 416))

        cv2.imshow('image', img)
        cv2.imshow('heatmaps', heatmaps[..., 9])
        cv2.imshow('masks', masks[..., 9])
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()
