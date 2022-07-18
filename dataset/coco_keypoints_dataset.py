# import sys
# import os
# sys.path.append(os.getcwd())
# from glob import glob

# import torch
# from torch.utils.data import Dataset, DataLoader
# import cv2
# import numpy as np
# import pytorch_lightning as pl
# import pycocotools
# from pycocotools.coco import COCO

# from dataset import transforms as T
# from dataset.keypoints_utils import HeatmapGenerator


# class COCOKeypointsDataset(Dataset):
#     def __init__(self, files_path, transforms, heatmap_generator, joints_generator, num_joints):
#         super().__init__()

#         self.coco = COCO(self._get_anno_file_name(files_path))
#         self.coco_imgs = self.coco.loadImgs(self.coco.getImgIds())
#         self.coco_imgs_name = [img['file_name'] for img in self.coco_imgs]
#         self.coco_imgs_id = [img['id'] for img in self.coco_imgs]
#         self.imgs = glob(files_path + '/*.jpg')
#         self.annos = self._get_annos(self.imgs)
#         self.num_joints = num_joints

#         self.transforms = transforms
#         self.heatmap_generator = heatmap_generator
#         self.joints_generator = joints_generator

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, index):
#         img_file = self.imgs[index]
#         img = cv2.imread(img_file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         anno = self.annos[index]
#         anno = [
#             obj for obj in anno
#             if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
#         ]

#         mask = self.get_mask(anno, img.shape[0], img.shape[1])

#         joints = self.get_joints(anno)

#         mask_list = [mask]
#         joints_list = [joints]

#         img, mask_list, joints_list = self.transforms(img, mask_list, joints_list)

#         target_t = self.heatmap_generator(joints_list[0])
#         joints_t = self.joints_generator(joints_list[0])
        
#         return img, target_t.astype(np.float32), mask_list[0].astype(np.float32), joints_t.astype(np.int32)

#     def _get_anno_file_name(self, files_path):
#         path_list = files_path.split(os.sep)
#         path_list[0] = os.sep
#         path = os.path.join(*path_list[:-1], 'annotations', f'person_keypoints_{os.path.basename(files_path)}.json')
#         return path

#     def _get_annos(self, imgs_list):
#         annos = []

#         for img_path in imgs_list:
#             file_name = os.path.basename(img_path)
#             img_id = self.coco_imgs_id[self.coco_imgs_name.index(file_name)]
#             ann_ids = self.coco.getAnnIds(img_id)
#             anns = self.coco.loadAnns(ann_ids)
#             annos.append(anns)

#         return annos
    
#     def get_joints(self, anno):
#         num_people = len(anno)

#         joints = np.zeros((num_people, self.num_joints, 3))

#         for i, obj in enumerate(anno):
#             joints[i, :self.num_joints, :3] = \
#                 np.array(obj['keypoints']).reshape([-1, 3])

#         return joints

#     def get_mask(self, anno, img_height, img_width):
#         m = np.zeros((img_height, img_width))

#         for obj in anno:
#             if obj['iscrowd']:
#                 rle = pycocotools.mask.frPyObjects(
#                     obj['segmentation'], img_height, img_width)
#                 m += pycocotools.mask.decode(rle)
#             elif obj['num_keypoints'] == 0:
#                 rles = pycocotools.mask.frPyObjects(
#                     obj['segmentation'], img_height, img_width)
#                 for rle in rles:
#                     m += pycocotools.mask.decode(rle)

#         return m < 0.5


# class COCOKeypointsDataModule(pl.LightningDataModule):
#     def __init__(self, cfg):
#         super().__init__()
#         self.train_path = cfg['train_path']
#         self.val_path = cfg['val_path']
#         self.workers = cfg['workers']
#         self.input_size = cfg['input_size']
#         self.output_size = cfg['output_size']
#         self.scale_type = cfg['scale_type']
#         self.batch_size = cfg['batch_size']
#         self.heatmap_generator = HeatmapGenerator(
#             cfg['output_size'], cfg['num_keypoints'], cfg['sigma']
#         )
#         self.num_joints = cfg['num_keypoints']
        
#     def setup(self, stage=None):
#         train_transforms = T.Compose([
#             T.RandomAffineTransform(
#                 input_size=self.input_size,
#                 output_size=self.output_size,
#                 max_rotation=30,
#                 min_scale=0.75,
#                 max_scale=1.5,
#                 scale_type=self.scale_type,
#                 max_translate=40
#             ),
#             T.ToTensor()
#         ])

#         valid_transform = T.Compose([
#             T.RandomAffineTransform(
#                 input_size=self.input_size,
#                 output_size=self.output_size,
#                 max_rotation=0,
#                 min_scale=1,
#                 max_scale=1,
#                 scale_type=self.scale_type,
#                 max_translate=0
#             ),
#             T.ToTensor()
#         ])
        
#         self.train_dataset = COCOKeypointsDataset(
#             self.train_path,
#             train_transforms, 
#             self.heatmap_generator,
#             self.joints_generator,
#             self.num_joints
#         )
        
#         self.valid_dataset = COCOKeypointsDataset(
#             self.val_path,
#             valid_transform, 
#             self.heatmap_generator,
#             self.joints_generator,
#             self.num_joints
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.workers,
#             persistent_workers=self.workers > 0,
#             pin_memory=self.workers > 0,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.valid_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.workers,
#             persistent_workers=self.workers > 0,
#             pin_memory=self.workers > 0,
#         )
