# import argparse
# import time
# import os

# import torch
# import numpy as np
# import cv2
# import timm

# from utils.yaml_helper import get_configs
# from module.yolov2_detector import YoloV2Detector
# from models.detector.yolov2 import YoloV2
# from dataset.detection.yolov2_utils import get_tagged_img, DecodeYoloV2, encode_target, decode_target, non_max_suppression
# from dataset.detection.yolov2_dataset import YoloV2DataModule
# from utils.module_select import get_model


# def inference(cfg, ckpt):
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

#     data_module = YoloV2DataModule(
#         train_list=cfg['train_list'], 
#         val_list=cfg['val_list'],
#         workers=cfg['workers'], 
#         input_size=cfg['input_size'],
#         batch_size=1
#     )
#     data_module.prepare_data()
#     data_module.setup()

#     backbone = get_model(cfg['backbone'])()
    
#     model = YoloV2(
#         backbone=backbone,
#         num_classes=cfg['num_classes'],
#         num_anchors=len(cfg['scaled_anchors'])
#     )

#     if torch.cuda.is_available:
#         model = model.cuda()
    
#     model_module = YoloV2Detector.load_from_checkpoint(
#         checkpoint_path=ckpt,
#         model=model, 
#         cfg=cfg
#     )
#     model_module.eval()

#     yolov2_decoder = DecodeYoloV2(cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'], conf_threshold=0.5)

#     # Inference
#     for sample in data_module.val_dataloader():
#         batch_x = sample['img']
#         batch_y = sample['annot']

#         if torch.cuda.is_available:
#             batch_x = batch_x.cuda()
        
#         before = time.time()
#         with torch.no_grad():
#             predictions = model_module(batch_x)
#         boxes = yolov2_decoder(predictions)
#         print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        
#         # batch_x to img
#         if torch.cuda.is_available:
#             img = batch_x.cpu()[0].numpy()   
#         else:
#             img = batch_x[0].numpy()   
#         img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         true_pred = encode_target(batch_y, cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'])
#         true_boxes = decode_target(true_pred, cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'])[0]
#         true_boxes = true_boxes[torch.where(true_boxes[..., 4] > 0)[0]]

#         tagged_img = get_tagged_img(img, boxes, cfg['names'], (0, 255, 0))
#         tagged_img = get_tagged_img(tagged_img, true_boxes, cfg['names'], (0, 0, 255))

#         cv2.imshow('test', tagged_img)
#         key = cv2.waitKey(0)
#         if key == 27:
#             break
    
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', required=True, type=str, help='config file')
#     parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
#     args = parser.parse_args()
#     cfg = get_configs(args.cfg)

#     inference(cfg, args.ckpt)
