import logging
import os
import sys
import traceback

from matplotlib import pyplot as plt
import requests

from inpainting.saicinpainting.evaluation.utils import move_to_device
from inpainting.saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
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


def load_image(fname, mode='RGB', return_orig=False):
    img = np.asarray(cv2.imread(fname, -1)[:, :, ::-1])
    img = (img / 2 ** 16).clip(0, 1).astype(np.float32)
    if img.ndim == 3:
        img = np.transpose(img, (2, 1, 0))  # HWC to CHW format
    img = torch.from_numpy(img)[None, ...]  # Add batch dimension
    print(img.shape)
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
        cur_res = cur_res.permute(2, 1, 0).cpu().numpy()  # Convert from CHW to HWC format
        cur_res = (cur_res * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite('inpainted_result.png', cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR))
        return cur_res
    
if __name__ == '__main__':
    # Load sample image from Hugging Face
    image_path = "/root/code/hdr/SIGGRAPH17_HDR_Trainingset/Training/002/262A0967.tif"
    img = load_image(image_path)
    # Generate random mask
    # Create mask with same shape as input image
    mask = torch.zeros(1, 1, img.shape[2], img.shape[3]).cuda()
    print(mask.shape)
    # Randomly generate some masked regions
    num_regions = np.random.randint(3, 8)  # Generate 3-7 random regions
    for _ in range(num_regions):
        # Random rectangle parameters
        h, w = img.shape[2], img.shape[3]
        x1 = np.random.randint(0, w-50)  # Ensure minimum 50px width
        y1 = np.random.randint(0, h-50)  # Ensure minimum 50px height
        width = np.random.randint(50, min(200, w-x1))  # Random width between 50-200px
        height = np.random.randint(50, min(200, h-y1))  # Random height between 50-200px
        mask[0, 0, y1:y1+height, x1:x1+width] = 1
    # Save mask image
    mask_np = mask[0,0].cpu().numpy() * 255
    mask_np = mask_np.transpose()
    mask_np = mask_np.astype(np.uint8)
    cv2.imwrite("generated_mask.png", mask_np)
 
    # Create colored visualization of inpainted region
    mask_vis = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    mask_vis[mask_np > 0] = [0, 0, 255]  # Set masked regions to red
    cv2.imwrite("inpaint_region_vis.png", mask_vis)
    # Pad image and mask to be divisible by 8
    def pad_to_multiple_of_8(tensor):
        h, w = tensor.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return tensor
    img =  pad_to_multiple_of_8(img)
    mask = pad_to_multiple_of_8(mask)
    get_inpaint(img, mask)