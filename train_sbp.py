import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from dataset.pose_mpii_dataset import SBPDataModule
from module.sbp_module import SBPDetector
from models.detector.sbp import SBP
from models.backbone.darknet import darknet19
from utils.utility import make_model_name
from utils.yaml_helper import get_configs


def train(cfg):
    data_module = SBPDataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        img_dir = cfg['img_dir'],
        input_size = cfg['input_size'],
        output_size = cfg['output_size'],
        num_keypoints = cfg['num_keypoints'],
        sigma = cfg['sigma'],
        workers = cfg['workers'],
        batch_size = cfg['batch_size'],
    )
    
    model = SBP(
        backbone_module_list=darknet19(pretrained='').get_features_module_list(), 
        num_keypoints=cfg['num_keypoints']
    )
    
    summary(model, (1, cfg['in_channels'], cfg['input_size'][0], cfg['input_size'][1]), device='cpu')

    model_module = SBPDetector(
        model=model, 
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True,
            every_n_epochs=cfg['save_freq']
        ),
        EarlyStopping(
            monitor='val_loss',
            # min_delta=0.00001,
            patience=40,
            verbose=True
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg), default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )
    
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)