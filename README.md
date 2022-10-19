# PyTorch Pose Estimation
PyTorch 기반 Pose Estimation 모델 구조 및 학습 기법을 테스트하기 위한 프로젝트

## Implementations
 * Simple Baselines for Human Pose Estimation
 
## TODOs
- [x] ~~Simple Baselines for Human Pose Estimation~~
- [ ] Single-Stage Multi-Person Pose Machines

## Requirements
* `PyTorch >= 1.8.1`
* `PyTorch Lightning`
* `Albumentations`
* `PyYaml`
* `Pycocotools`

## Train Detector
```python
python train_sbp.py --cfg configs/sbp_coco.yaml
```

## Test Detector
```python
python test_sbp.py --cfg configs/sbp_coco.yaml
```

## Inference Detector
```python
python inference_sbp.py --cfg configs/sbp_coco.yaml --ckpt path/to/ckpt_file
```

## 데이터셋
| Train Dataset | Validation Dataset |
| --- | --- |
| COCO2017 Train | COCO2017 Validation |

## 결과
| Method | Backbone | Input size | AP@.5 |
| --- | --- | --- | --- |
| Simple Baselines | darknet19 | 256 x 192 | 80.9 |
