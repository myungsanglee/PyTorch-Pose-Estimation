import sys
import os
import copy
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from pycocotools.coco import COCO

from dataset.keypoints_utils import SPMHeatmapGenerator, SPMDisplacementGenerator, SPMMaskGenerator, DecodeSPM
from utils.yaml_helper import get_configs


class SPMCOCODataset(Dataset):
    def __init__(self, img_dir, file_path, transforms, heatmap_generator, displacement_generator, mask_generator, ratio, num_keypoints):
        super().__init__()

        self.img_dir = self._get_img_dir(img_dir, file_path)
        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.displacement_generator = displacement_generator
        self.mask_generator = mask_generator
        self.ratio = ratio # output_size / input_size
        # self.class_labels = np.array(class_labels)
        self.num_keypoints = num_keypoints
        
        self.coco = COCO(file_path)
        
        self.cats_dict = dict([[cat['id'], cat['name']] for cat in self.coco.loadCats(self.coco.getCatIds())])
             
        self.image_set_index = self.coco.getImgIds()
        
        self.db = self._load_coco_keypoint_annotations()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        db_rec = copy.deepcopy(self.db[index])

        img = cv2.imread(db_rec['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        joints = db_rec['joints'] # (num_person, num_keypoints, 2)
        joints_vis = db_rec['joints_vis'] # (num_person, num_keypoints)
        centers = db_rec['centers'] # (num_person, 2)
        
        # concat joints & centers
        keypoints = np.concatenate([joints, np.reshape(centers, (-1, 1, 2))], axis=1) # (num_person, (num_keypoints + 1), 2)
        
        # convert keypoints shape to (-1, 2)
        keypoints = np.reshape(keypoints, (-1, 2)) # (num_person * (num_keypoints + 1), 2)
        
        # transform
        transformed = self.transforms(image=img, keypoints=keypoints)
        transformed_img = transformed['image']
        transformed_keypoints = np.array(transformed['keypoints']) # (num_person * (num_keypoints + 1), 2)
        
        # convert image [height, width, channel] to [channel, height, width]
        transformed_img = np.transpose(transformed_img, (2, 0, 1))
        
        # convert 'keypoints coordinates' input_size ratio to ouput_size ratio
        keypoints = transformed_keypoints * self.ratio
        
        # divide keypoints to centers & joints
        # keypoints = np.reshape(keypoints, (-1, self.num_keypoints+1, 2))
        keypoints = np.reshape(keypoints, (-1, self.num_keypoints+1, 2)).astype(np.int)
        centers = keypoints[:, self.num_keypoints:, :] # (num_person, 1, 2)
        joints = keypoints[:, :self.num_keypoints, :] # (num_person, num_keypoints, 2)
        
        # get heatmaps of root joints
        heatmaps = self.heatmap_generator(centers) # (1, output_h, output_w)
        
        # get masks
        masks = self.mask_generator(centers)
        
        # get displacement maps
        displacements = self.displacement_generator(joints, masks) # (2*num_keypoints, output_h, output_w)
        
        # concat heatmaps & displacements
        target = np.concatenate([heatmaps, displacements], axis=0) # (1 + 2*(num_keypoints), output_h, output_w)
        
        return transformed_img, target

    def _get_img_dir(self, img_dir, file_path):
        img_dir_name = os.path.splitext(file_path.split('_')[-1])[0]
        
        return os.path.join(img_dir, img_dir_name)

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db
    
    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        file_name = im_ann['file_name']
        img_id = im_ann['id']
        
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        tmp_joints = []
        tmp_joints_vis = []
        tmp_centers = []
        for obj in objs:
            cls = self.cats_dict[obj['category_id']]
            if cls != 'person':
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            x1 = int(obj['clean_bbox'][0])
            y1 = int(obj['clean_bbox'][1])
            x2 = x1 + int(obj['clean_bbox'][2])
            y2 = y1 + int(obj['clean_bbox'][3])
            
            cx = (x1 + x2) / 2.
            cy = (y1 + y2) / 2.

            joints = np.zeros((self.num_keypoints, 2), dtype=np.float)
            joints_vis = np.zeros((self.num_keypoints), dtype=np.float)
            for ipt in range(self.num_keypoints):
                if x1 < obj['keypoints'][ipt * 3 + 0] < x2 and y1 < obj['keypoints'][ipt * 3 + 1] < y2:                
                    joints[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_vis[ipt] = t_vis

            if np.sum(joints_vis) == 0:
                continue

            tmp_joints.append(joints)
            tmp_joints_vis.append(joints_vis)
            tmp_centers.append([cx, cy])

        if tmp_joints:
            rec.append({
                'image_path': os.path.join(self.img_dir, file_name),
                'joints': np.array(tmp_joints),
                'joints_vis': np.array(tmp_joints_vis),
                'centers': np.array(tmp_centers),
                'image_id': img_id,
                'category_id': obj['category_id'],
            })

        return rec


class SPMCOCODataModule(pl.LightningDataModule):
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
        # class_labels
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
        self.heatmap_generator = SPMHeatmapGenerator(
            output_size, 1, sigma
        )
        self.displacement_generator = SPMDisplacementGenerator(
            output_size, num_keypoints
        )
        self.mask_generator = SPMMaskGenerator(
            output_size, sigma
        )
        self.ratio = self.output_size / self.input_size
        # self.class_labels = class_labels
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            # A.Rotate(limit=40),
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            # A.RandomResizedCrop(self.input_size, self.input_size, (0.5, 1), (0.4, 1.6)),
            A.Resize(self.input_size, self.input_size),
            A.Normalize(0, 1)
        # ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
        ], keypoint_params=A.KeypointParams(format='xy'))

        valid_transform = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(0, 1)
        # ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
        ], keypoint_params=A.KeypointParams(format='xy'))
        
        self.train_dataset = SPMCOCODataset(
            self.img_dir,
            self.train_path,
            train_transforms, 
            self.heatmap_generator,
            self.displacement_generator,
            self.mask_generator,
            self.ratio,
            # self.class_labels,
            self.num_keypoints
        )
        
        self.valid_dataset = SPMCOCODataset(
            self.img_dir,
            self.val_path,
            valid_transform, 
            self.heatmap_generator,
            self.displacement_generator,
            self.mask_generator,
            self.ratio,
            # self.class_labels,
            self.num_keypoints
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
    from dataset.keypoints_utils import MeanAveragePrecision
    cfg = get_configs('./configs/spm_coco.yaml')

    data_module = SPMCOCODataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        img_dir = cfg['img_dir'],
        input_size = cfg['input_size'],
        output_size = cfg['output_size'],
        num_keypoints = cfg['num_keypoints'],
        sigma = cfg['sigma'],
        workers = cfg['workers'],
        batch_size = 1
        # batch_size = cfg['batch_size'],
        # class_labels=cfg['class_labels']
    )
    data_module.prepare_data()
    data_module.setup()

    spm_decoder = DecodeSPM(cfg['input_size'], cfg['sigma'], 0.99, False)
    
    # map_metric = MeanAveragePrecision(cfg['val_path'], cfg['input_size'], cfg['conf_threshold'])
    # map_metric.reset_states()

    # for img, target in data_module.train_dataloader():
    for img, target in data_module.val_dataloader():        
        # convert img to opencv numpy array
        img = img[0].permute((1, 2, 0)).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        heatmaps = target[0, 0, :, :].numpy()
        
        root_joints, keypoints_joint = spm_decoder(target)
        # map_metric.update_state(target, heatmaps)
        
        # Draw Root joints
        for x, y in root_joints:
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # Draw keypoints joint
        for joints in keypoints_joint:
            for x, y in joints:
                if x == 0. and y == 0.:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        
        # # Draw keypoints joint
        # for idx, (x, y, conf) in enumerate(joints):
        #     if conf < 0:
        #         continue
        #     x, y = int(x), int(y)
        #     cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            
        #     margin = 10
        #     org_x = np.clip(x, margin, cfg['input_size'][1] - margin)
        #     org_y = np.clip(y, margin, cfg['input_size'][0] - margin)
        #     cv2.putText(img, f'{idx}({conf:.2f})', (org_x, org_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        
        # heatmaps = heatmaps[0].permute((1, 2, 0)).numpy()
        # heatmaps = np.sum(heatmaps, axis=-1)
        # heatmaps = cv2.resize(heatmaps, (192, 256))
        
        cv2.imshow('image', img)
        cv2.imshow('heatmaps', heatmaps)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

    # map = map_metric.result()
    # print(map)