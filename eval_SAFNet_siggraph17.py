import os
import shutil
import time
import math
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from models.SAFNet import SAFNet
from dataset.datasets import HDRDataset
from utils.utils import prepare_input_images, range_compressor, calculate_psnr
import torch.nn.functional as F
import matplotlib.pyplot as plt

dataset_eval = HDRDataset(root_dir='/root/code/hdr/kata17',is_train=False)
dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=1)
model = SAFNet().cuda().eval()
model.load_state_dict(torch.load('/root/code/hdr/SAFNet/checkpoints/SAFNet_epoch_1980.pth'))

test_results_path = './img_hdr_pred_siggraph17'

if os.path.exists(test_results_path):
    shutil.rmtree(test_results_path)
    
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)

psnr_l_list = []
psnr_m_list = []

for i, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(dataloader_eval):
    img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, "cuda")
    inf = torch.cat([img0_c,img1_c,img2_c],1)
    print(inf.shape)
    # Pad the width dimension from 1500 to 1504

    with torch.no_grad():
        img_hdr_m,mask0, mask2 = model.forward_mask( img0_c, img1_c, img2_c)
    psnr_l = calculate_psnr(img_hdr_m.cpu(), img_hdr_gt.cpu()).cpu()

    img_hdr_r_m = range_compressor(img_hdr_m)
    img_hdr_gt_m = range_compressor(img_hdr_gt)
    psnr_m = calculate_psnr(img_hdr_r_m.cpu(), img_hdr_gt_m.cpu()).cpu()

    psnr_l_list.append(psnr_l)
    psnr_m_list.append(psnr_m)

    print('SIGGRAPH17 Test {:03d}: PSNR_l={:.3f} PSNR_m={:.3f}'.format(i+1, psnr_l, psnr_m))

 
    img_hdr_r_np = img_hdr_m[0].data.cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(test_results_path, '{:03d}.hdr'.format(i+1)), img_hdr_r_np[:, :, ::-1])
    
    # Convert masks to histograms
    mask0_np = mask0[0,0].data.cpu().numpy().flatten()  # Flatten to 1D array
    mask2_np = mask2[0,0].data.cpu().numpy().flatten()
    
    # Create histograms
    plt.figure(figsize=(10, 4))
    
    # Plot mask0 histogram
    plt.subplot(1, 2, 1)
    plt.hist(mask0_np, bins=50, range=(0, 1), density=True)
    plt.title('Mask0 Value Distribution')
    plt.xlabel('Mask Value')
    plt.ylabel('Density')
    
    # Plot mask2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(mask2_np, bins=50, range=(0, 1), density=True)
    plt.title('Mask2 Value Distribution')
    plt.xlabel('Mask Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_results_path, '{:03d}_mask_dist.png'.format(i+1)))
    plt.close()
print('SIGGRAPH17 Test Average: PSNR_l={:.3f} PSNR_m={:.3f}'.format(np.array(psnr_l_list).mean(), np.array(psnr_m_list).mean()))
