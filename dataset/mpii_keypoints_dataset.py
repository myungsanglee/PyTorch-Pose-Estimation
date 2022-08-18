import sys
import os
sys.path.append(os.getcwd())
from glob import glob
import json
import math

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations as A

from dataset.keypoints_utils import HeatmapGenerator, DisplacementGenerator, MaskGenerator, DecodeSPM


class MPIIKeypointsDataset(Dataset):
    def __init__(self, img_dir, file_path, transforms, heatmap_generator, displacement_generator, mask_generator, ratio):
        super().__init__()

        self.img_dir = img_dir
        self.data_dict = self._get_data_dict(file_path)
        self.imgs = list(self.data_dict.keys())
        # print(f'Dataset Num: {len(self.imgs)}')

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.displacement_generator = displacement_generator
        self.mask_generator = mask_generator
        self.ratio = ratio # output_size / input_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # get image 
        img_filename = self.imgs[index]
        img_file = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # get annotations
        anno = self.data_dict[img_filename]
        joints = anno['joints']
        centers = anno['centers']

        # combine joints in one list
        keypoints = []
        for center, joint in zip(centers, joints):
            joint.append(center)
            for idx, [x, y] in enumerate(joint):
                if x >= w:
                    x = w - 1
                if y >= h:
                    y = h -1
                joint[idx] = [x, y]
            keypoints += joint

        # transform
        transformed = self.transforms(image=img, keypoints=keypoints)
        # return transformed

        transformed_img = transformed['image']
        transformed_keypoints = transformed['keypoints']

        # convert image [height, width, channel] to [channel, height, width]
        img = np.transpose(transformed_img, (2, 0, 1))

        # parsing keypoints to joints & centers
        joints, centers = self._parsing_keypoints(transformed_keypoints)

        # get heatmaps of root joints
        heatmaps = self.heatmap_generator(centers)

        # get masks
        masks = self.mask_generator(centers)

        # get displacement maps
        displacements = self.displacement_generator(joints, masks)

        # concat heatmaps & displacements
        target = np.concatenate([heatmaps, displacements], axis=0)

        return img, target

    def _get_data_dict(self, files_path):
        data_dict = dict()
        with open(files_path, 'r') as f:
            data = json.load(f)
        
        for anno in data:
            image = anno['image']
            try:
                tmp_anno = data_dict[image]
                tmp_anno['joints'].append([[int(x), int(y)] if x > 0 and y > 0 else [0, 0] for x, y in anno['joints']])
                tmp_anno['centers'].append([int(anno['center'][0]), int(anno['center'][1])])

            except:
                data_dict[image] = {
                    'joints': [[[int(x), int(y)] if x > 0 and y > 0 else [0, 0] for x, y in anno['joints']]],
                    'centers': [[int(anno['center'][0]), int(anno['center'][1])]]
                }

        return data_dict
    
    def _parsing_keypoints(self, keypoints):
        # parsing keypoints 
        joints = []
        centers = []
        tmp_joint = []
        tmp_center = []
        tmp_idx = 0
        for (x, y) in keypoints:
            if tmp_idx > 16:
                tmp_idx = 0

            # Joints
            if tmp_idx < 16:
                # convert 'Point' input_size ratio to ouput_size ratio
                x, y = int(x*self.ratio), int(y*self.ratio)
                tmp_joint.append([x, y])

            # Center
            else:
                # convert 'Point' input_size ratio to ouput_size ratio
                x, y = int(x*self.ratio), int(y*self.ratio)
                tmp_center.append([x, y])

                # append data & initialize
                joints.append(tmp_joint)
                centers.append(tmp_center)
                tmp_joint = []
                tmp_center = []

            tmp_idx += 1

        return joints, centers

class MPIIKeypointsDataModule(pl.LightningDataModule):
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
        self.batch_size = batch_size
        self.heatmap_generator = HeatmapGenerator(
            output_size, 1, sigma
        )
        self.displacement_generator = DisplacementGenerator(
            output_size, num_keypoints
        )
        self.mask_generator = MaskGenerator(
            output_size, sigma
        )
        self.ratio = self.output_size / self.input_size
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            A.Resize(self.input_size, self.input_size),
            A.Normalize(0, 1)
        ], keypoint_params=A.KeypointParams(format='xy'))

        valid_transform = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(0, 1)
        ], keypoint_params=A.KeypointParams(format='xy'))
        
        self.train_dataset = MPIIKeypointsDataset(
            self.img_dir,
            self.train_path,
            train_transforms, 
            self.heatmap_generator,
            self.displacement_generator,
            self.mask_generator,
            self.ratio
        )
        
        self.valid_dataset = MPIIKeypointsDataset(
            self.img_dir,
            self.val_path,
            valid_transform, 
            self.heatmap_generator,
            self.displacement_generator,
            self.mask_generator,
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
    cfg = dict()

    # cfg['train_path'] = '/home/fssv2/myungsang/datasets/mpii_human_pose/annotations/train.json'
    # cfg['val_path'] = '/home/fssv2/myungsang/datasets/mpii_human_pose/annotations/valid.json'
    cfg['train_path'] = '/home/fssv2/myungsang/datasets/mpii_human_pose/annotations/tmp.json'
    cfg['val_path'] = '/home/fssv2/myungsang/datasets/mpii_human_pose/annotations/tmp.json'
    cfg['img_dir'] = '/home/fssv2/myungsang/datasets/mpii_human_pose/images'
    cfg['class_labels'] = [
        'r_ankle', 
        'r_knee', 
        'r_hip', 
        'l_hip', 
        'l_knee', 
        'l_ankle', 
        'pelvis', 
        'thorax', 
        'upper_neck', 
        'head_top', 
        'r_wrist', 
        'r_elbow', 
        'r_shoulder',
        'l_shoulder', 
        'l_elbow',
        'l_wrist'
    ]
    cfg['workers'] = 0
    cfg['input_size'] = 512
    cfg['output_size'] = 64
    cfg['batch_size'] = 1
    cfg['num_keypoints'] = 16
    cfg['sigma'] = 2
    
    data_module = MPIIKeypointsDataModule(
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

    spm_decoder = DecodeSPM(cfg['input_size'], cfg['sigma'], 0.99, False)

    for img, target in data_module.train_dataloader():
    # for img, target in data_module.val_dataloader():
        # print(type(img), img.size())
        # print(type(target), target.size())

        # continue
        
        # convert img to opencv numpy array
        img = img[0].permute((1, 2, 0)).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        heatmaps = target[0, 0, :, :].numpy()

        root_joints, keypoints_joint = spm_decoder(target)

        # Draw Root joints
        for x, y in root_joints:
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # Draw keypoints joint
        for joints in keypoints_joint:
            for x, y in joints:
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

        # heatmaps = cv2.resize(heatmaps, (416, 416))

        cv2.imshow('image', img)
        cv2.imshow('heatmaps', heatmaps)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()
