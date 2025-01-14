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
        print(f"Loading pretrained model from {config.model['pretrained_path']}")
        state_dict = torch.load(config.model['pretrained_path'])
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        missing_keys = [k for k in new_state_dict if k not in filtered_state_dict]
        unexpected_keys = [k for k in model_state_dict if k not in new_state_dict]
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
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
    
    elif config.scheduler['name'] == 'LinearLR':
        try:
            scheduler_params = config.scheduler.get('params', {})
            if not isinstance(scheduler_params, dict):
                scheduler_params = {}
            
            # Set default values
            default_params = {
                'start_factor': 1.0,
                'end_factor': 0.1,
                'total_iters': int(config.train['num_epochs'] * steps_per_epoch)
            }
            
            # Update defaults with provided params
            for key, value in scheduler_params.items():
                if key in ['start_factor', 'end_factor']:
                    default_params[key] = float(value)
            print(f"Scheduler params: {default_params}")  # Debug print
            
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                **default_params
            )
        except Exception as e:
            print(f"Error in scheduler configuration: {e}")
            print(f"Config scheduler section: {config.scheduler}")
            raise

def get_dataloader(dataset_dir, config, is_train=True):
    # 创建数据集
    dataset = HDRDataset(
        dataset_dir, 
        patch_size=config.data['patch_size'], 
        is_train=is_train
    )
    
    return DataLoader(
        dataset,
        batch_size=config.data['batch_size'] if is_train else 1,
        shuffle=is_train,                   
        num_workers=config.data['num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False                     
    )