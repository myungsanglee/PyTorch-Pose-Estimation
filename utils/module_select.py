from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from module.lr_scheduler import CosineAnnealingWarmUpRestarts, YoloLR


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {
        'sgd': optim.SGD, 
        'adam': optim.Adam,
        'radam': optim.RAdam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop
    }
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)


def get_scheduler(scheduler_name, optim, **kwargs):
    scheduler_dict = {
        'multi_step': MultiStepLR, 
        'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
        'cosine_annealing_warm_up_restarts': CosineAnnealingWarmUpRestarts,
        'yolo_lr': YoloLR
    }
    optimizer = scheduler_dict.get(scheduler_name)
    if optimizer:
        return optimizer(optim, **kwargs)
