import os
import math
import json
from tqdm import tqdm

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == '__main__':
    coco = COCO('/home/fssv2/myungsang/datasets/coco_2017/tmp_keypoints/annotations/tmp_person_keypoints_val2017.json')
    # coco = COCO('/home/fssv2/myungsang/datasets/pis/coco_format/pis_person_keypoints_train.json')
    # coco = COCO('/home/fssv2/myungsang/datasets/pis/coco_format/pis_person_keypoints_valid.json')
    # coco = COCO(val_path)
    
    imgs = coco.loadImgs(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())

    imgs_info = [[img['id'], img['file_name'], img['width'], img['height']] for img in imgs]
    cats_dict = dict([[cat['name'], cat['id']] for cat in cats])
    
    results = []
    results_json_path = os.path.join(os.getcwd(), 'results.json')

    # Inference
    for (img_id, img_name, width, height) in tqdm(imgs_info):
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            cat_id = ann['category_id']
            keypoints = ann['keypoints']
            
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "keypoints": keypoints,
                "score": float(0.9)
            })
            
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)


    img_ids=sorted(coco.getImgIds())
    cat_ids=sorted(coco.getCatIds())

    # load detection JSON file from the disk
    cocovalPrediction = coco.loadRes(results_json_path)
	# initialize the COCOeval object by passing the coco object with
	# ground truth annotations, coco object with detection results
    cocoEval = COCOeval(coco, cocovalPrediction, "keypoints")
	
	# run evaluation for each image, accumulates per image results
	# display the summary metrics of the evaluation
    cocoEval.params.imgIds  = img_ids
    cocoEval.params.catIds  = cat_ids
 
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    
    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, cocoEval.stats[ind]))
    
    print(info_str)

