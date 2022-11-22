import os
import math
import json
from tqdm import tqdm
from collections import OrderedDict

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.detector.sbp import SBP
from utils.module_select import get_model    


'''
save weights only specific layers
'''


backbone_features_module = get_model('darknet19')(
        pretrained='',
        features_only=True
    )

model = SBP(
    backbone_features_module=backbone_features_module,
    num_keypoints=17
)

ckpt_path = os.path.join(os.getcwd(), 'saved/simple-baselines-pose_coco-keypoints/version_0/checkpoints/epoch=194-step=113879.ckpt')
checkpoint = torch.load(ckpt_path)
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for key in list(state_dict):
    layer_name = key.split('.')[1]
    if layer_name != 'backbone_features_module':
        break
    new_state_dict[key.replace("model.", "")] = state_dict.pop(key)

torch.save(new_state_dict, 'pretrained_weights.pt')
# new_state_dict = torch.load(os.path.join(os.getcwd(), 'saved/yolov4-tiny_coco-person/version_0/checkpoints/pretrained_weights.pt'))

# model.load_state_dict(new_state_dict, False)

# Check param values
# for name1, m1 in model.named_children():
#     print(name1, m1)
#     for name2, m2 in m1.named_children():
#         print(name2, m2)
#         for param in m2.parameters():
#             print(param[0, 0, 0, :])
#             break
#         break
#     break