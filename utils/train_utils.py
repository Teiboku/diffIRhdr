import torch
import wandb
from models.NHDR import NHDRRNet
from models.SAFNet import SAFNet
from models.UNet import  Unet
from models.archs.NAFNet_arch import NAFNet
from dataset.datasets import HDRDataset
from torch.utils.data import DataLoader


def init_wandb(config):
    wandb.init(
        project=config.logging['wandb']['project'],
        entity=config.logging['wandb']['entity'],
        tags=config.logging['wandb']['tags'],
        config=config.config
    )

def get_model(config):
    if config.model['name'] == 'NAFNet':
        model = NAFNet(**config.model['params'])
    elif config.model['name'] == 'SAFNet':
        model = SAFNet(**config.model['params'])
    elif config.model['name'] == 'Unet':
        model = Unet(config.model['params'])
    elif config.model['name'] == 'NHDR':
        model = NHDRRNet(config.model['params'])
    else:
        raise ValueError(f"Unsupported model: {config.model['name']}")
        
    if config.model['pretrained_path']:
        model.load_state_dict(torch.load(config.model['pretrained_path']))
    return model

def get_optimizer(model, config):
    if config.optimizer['name'] == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.optimizer['lr']),
            weight_decay=float(config.optimizer['weight_decay']),
            betas=tuple(map(float, config.optimizer['betas']))
        )
    elif config.optimizer['name'] == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(), 
            lr=float(config.optimizer['lr']),
            weight_decay=float(config.optimizer['weight_decay']),
            betas=tuple(map(float, config.optimizer['betas']))
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer['name']}")

def get_scheduler(optimizer, config, steps_per_epoch):
    if config.scheduler['name'] == 'OneCycleLR':
        scheduler_params = config.scheduler['params'].copy()
        total_steps = int(config.train['num_epochs']) * steps_per_epoch
        scheduler_params['total_steps'] = total_steps
        
        numeric_params = ['max_lr', 'div_factor', 'final_div_factor']
        for param in numeric_params:
            if param in scheduler_params:
                scheduler_params[param] = float(scheduler_params[param])
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    
def get_dataloader(dataset_dir, config, is_train=True):
    """统一的数据加载器创建函数"""
    dataset = HDRDataset(dataset_dir, patch_size=config.data['patch_size'], is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=config.data['batch_size'] if is_train else 1,
        num_workers=config.data['num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
