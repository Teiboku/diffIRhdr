import os
import shutil
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from models.SAFNet import SAFNet
from dataset.datasets import HDRDataset
from utils.utils import prepare_input_images, range_compressor, calculate_psnr, generate_diff_map, tone_mapping

# 初始化
test_results_path = './img_hdr_pred_siggraph17'
if os.path.exists(test_results_path):
    shutil.rmtree(test_results_path)
os.makedirs(test_results_path)

# 加载数据和模型
dataset_eval = HDRDataset(root_dir='./dataset/ka17', is_train=False)
dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=1)
model = SAFNet().cuda().eval()
state_dict = torch.load('./checkpoints/sota.pth')
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})

psnr_l_list, psnr_m_list = [], []

# 评估循环
for i, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(dataloader_eval):
    # 准备输入
    img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, "cuda")
    
    # 模型推理
    with torch.no_grad():
        img_hdr_m = model(img0_c, img1_c, img2_c)
    
    # 计算PSNR
    psnr_l = calculate_psnr(img_hdr_m.cpu(), img_hdr_gt.cpu()).cpu()
    img_hdr_r_m = range_compressor(img_hdr_m)
    img_hdr_gt_m = range_compressor(img_hdr_gt)
    psnr_m = calculate_psnr(img_hdr_r_m.cpu(), img_hdr_gt_m.cpu()).cpu()
    psnr_l_list.append(psnr_l)
    psnr_m_list.append(psnr_m)
    
    print(f'SIGGRAPH17 Test {i+1:03d}: PSNR_l={psnr_l:.3f} PSNR_m={psnr_m:.3f}')
    
    # 保存结果
    # 差异图
    diff_map = generate_diff_map(img_hdr_r_m, img_hdr_gt_m)
    diff_map_bw = 255 - (diff_map.data.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(test_results_path, f'{i+1:03d}_diff_bw.png'), diff_map_bw)
    
    # HDR和LDR图像
    img_hdr_r_np = img_hdr_m.data.cpu().squeeze(0).permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(test_results_path, f'{i+1:03d}.hdr'), img_hdr_r_np[:, :, ::-1])
    cv2.imwrite(os.path.join(test_results_path, f'{i+1:03d}.png'), 
                (tone_mapping(img_hdr_r_np)[:, :, ::-1] * 255).astype(np.uint8))
    
    # Ground truth
    img_hdr_gt_np = img_hdr_gt.data.cpu().squeeze(0).permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(test_results_path, f'{i+1:03d}_gt.png'),
                (tone_mapping(img_hdr_gt_np)[:, :, ::-1] * 255).astype(np.uint8))

# 打印平均PSNR
print(f'SIGGRAPH17 Test Average: PSNR_l={np.array(psnr_l_list).mean():.3f} PSNR_m={np.array(psnr_m_list).mean():.3f}')
