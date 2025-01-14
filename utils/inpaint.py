import logging
import os
import sys
import traceback

from matplotlib import pyplot as plt
import requests

# Add project root path to python path
import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add project root to python path if not already there
if project_root not in sys.path:
    sys.path.append(project_root)




os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import PIL.Image as Image
from inpainting.saicinpainting.training.data.datasets import make_default_val_dataset
from inpainting.saicinpainting.training.trainers import load_checkpoint
from inpainting.saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')


def load_image(fname, mode='RGB', return_orig=False):
    # 读取PNG图像
    img = cv2.imread(fname)
    if img is None:
        raise ValueError(f"Failed to load image: {fname}")
    
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化到[0,1]
    img = img.astype(np.float32) / 255.0
    
    # HWC转CHW格式
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
        
    # 添加batch维度
    img = torch.from_numpy(img)[None, ...]
    
    print(f"Loaded image shape: {img.shape}")
    return img


def get_inpaint(image: torch.Tensor, mask: torch.Tensor,unpad_to_size: tuple = None,predict_config_path: str = '/root/code/hdr/SAFNet/configs/default.yaml'):
    with open(predict_config_path, 'r') as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))
    print(f'predict_config: {predict_config}')
    device = torch.device('cuda')
    print(f'predict_config.model.path: {predict_config.model.path}')
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    # train_config.visualizer.kind = 'noop'
    out_ext = ('.png')
    checkpoint_path = os.path.join(predict_config.model.path, 
                                    'models', 
                                    "last.ckpt")
    print(f'checkpoint_path: {checkpoint_path}')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cuda')
    
    model.freeze()
    model.to(device)
    if not predict_config.indir.endswith('/'):
        predict_config.indir += '/'
    with torch.no_grad():
        batch = {}
        batch['mask'] = mask
        batch['image'] = image
        batch = move_to_device(batch, device)
        batch = model(batch)   
        cur_res = batch[predict_config.out_key][0]
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
        # Save the inpainted result
        cur_res = cur_res.permute(1, 2, 0)  # Convert from CHW (3,1000,1504) to HWC (1000,1504,3) format
        cur_res = cur_res.cpu().numpy()
        cur_res = (cur_res * 255).clip(0, 255).astype(np.uint8)
        print(f'cur_res: {cur_res.shape}')
        cv2.imwrite('inpainted_result.png', cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR))
        return cur_res
    
if __name__ == '__main__':
    # Load sample image from Hugging Face
    image_path = "/root/autodl-tmp/diffIRhdr/wandb/latest-run/files/media/images/predicted_image_17002_91f75703ce28e8c4240a.png"
    gt_path = "/root/autodl-tmp/diffIRhdr/wandb/latest-run/files/media/images/ground_truth_image_17002_d956d6271e818c6ad1bd.png"
    img = load_image(image_path)
    gt = load_image(gt_path)
    
    # Pad dimensions to be divisible by 8
    h, w = img.shape[2], img.shape[3]
    new_h = ((h + 7) // 8) * 8  # Round up to nearest multiple of 8
    new_w = ((w + 7) // 8) * 8
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    # Pad the images
    img = F.pad(img, (0, pad_w, 0, pad_h))
    gt = F.pad(gt, (0, pad_w, 0, pad_h))
    mask = torch.zeros(1, 1, new_h, new_w)
    # Calculate absolute difference between gt and img
    diff = torch.abs(gt - img)
    
    # Calculate mean difference across color channels
    mean_diff = torch.mean(diff, dim=1, keepdim=True)
    # Calculate threshold as a percentile of the differences
    threshold = torch.quantile(mean_diff, 0.95)  # Top 20% of differences will be masked
    mask = (mean_diff > threshold).float()
    # Save mask heatmap
    mask_np = mask[0,0].cpu().numpy()  # Convert to numpy array and remove batch/channel dimensions
    plt.figure(figsize=(10,10))
    plt.imshow(mask_np, cmap='hot')
    plt.colorbar()
    plt.savefig('mask_heatmap.png')
    plt.close()
    
    res = get_inpaint(img, mask)
    # Convert res back to tensor format and normalize to [0,1]
    res_tensor = torch.from_numpy(res).permute(2,0,1).unsqueeze(0).float() / 255.0
    
    # Calculate MSE loss between res and gt
    mse_res = F.mse_loss(res_tensor, gt)
    # Calculate MSE loss between img and gt 
    mse_img = F.mse_loss(img, gt)
    
    # Calculate PSNR
    psnr_res = -10 * torch.log10(mse_res)
    psnr_img = -10 * torch.log10(mse_img)
    
    print(f'MSE loss between inpainted result and ground truth: {mse_res:.6f}')
    print(f'MSE loss between original image and ground truth: {mse_img:.6f}')
    print(f'PSNR between inpainted result and ground truth: {psnr_res:.2f} dB')
    print(f'PSNR between original image and ground truth: {psnr_img:.2f} dB')