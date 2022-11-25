import argparse
import platform

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from dataset.sbp_pis_dataset import SBPPISDataModule
from module.sbp_pis_detector import SBPPISDetector
from models.detector.sbp import SBP
from utils.module_select import get_model
from utils.utility import make_model_name
from utils.yaml_helper import get_configs


def train(cfg):
    data_module = SBPPISDataModule(
        train_path = cfg['train_path'],
        val_path = cfg['val_path'],
        input_size = cfg['input_size'],
        output_size = cfg['output_size'],
        num_keypoints = cfg['num_keypoints'],
        sigma = cfg['sigma'],
        workers = cfg['workers'],
        batch_size = cfg['batch_size'],
        class_labels=cfg['class_labels']
    )
    
    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'],
        features_only=True
    )
    
    model = SBP(
        backbone_features_module=backbone_features_module, 
        num_keypoints=cfg['num_keypoints']
    )
    
    if cfg['model_pretrained']:
        state_dict = torch.load(cfg['model_pretrained'])
        model.load_state_dict(state_dict, False)
    
    summary(model, (1, cfg['in_channels'], cfg['input_size'][0], cfg['input_size'][1]), device='cpu')

    model_module = SBPPISDetector(
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
            patience=30,
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
