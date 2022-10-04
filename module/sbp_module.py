import pytorch_lightning as pl

from utils.module_select import get_optimizer, get_scheduler
from models.loss.sbp_loss import SBPLoss
from dataset.keypoints_utils import MeanAveragePrecision


class SBPDetector(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = SBPLoss()
        self.map_metric = MeanAveragePrecision(cfg['val_path'], cfg['input_size'], cfg['conf_threshold'])

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self.model(img)
        
        loss = self.loss_fn(pred, target['heatmaps'])

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_start(self):
        self.map_metric.reset_states()

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self.model(img)

        loss = self.loss_fn(pred, target['heatmaps'])

        self.log('val_loss', loss, prog_bar=True, logger=True)

        self.map_metric.update_state(target, pred)        

    def on_validation_epoch_end(self):
        map = self.map_metric.result()
        self.log('val_mAP', map, prog_bar=True, logger=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )

            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            } 
        
        except KeyError:
            return optim
