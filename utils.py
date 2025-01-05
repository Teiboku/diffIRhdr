import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision.models.optical_flow import raft_large
import wandb

from mask_util.hdr_mask import get_diff_mask

def prepare_masks(imgs_ldr,device):
    img0 = imgs_ldr[0]
    img1 = imgs_ldr[1]
    img2 = imgs_ldr[2]
    mask0 = get_diff_mask(img0,img1).to(device)
    mask2 = get_diff_mask(img2,img1).to(device)
    return mask0,mask2

def prepare_input_images(imgs_lin, imgs_ldr, device):
    img0_c = torch.cat([imgs_lin[0], imgs_ldr[0]], 1).to(device)
    img1_c = torch.cat([imgs_lin[1], imgs_ldr[1]], 1).to(device) 
    img2_c = torch.cat([imgs_lin[2], imgs_ldr[2]], 1).to(device)
    return img0_c, img1_c, img2_c

def visualize_weights(weight_vis, output_dir, epoch, index, dpi=300):
    weight_vis_np = weight_vis[0].cpu().numpy()
    weight_vis_np = np.transpose(weight_vis_np, (1,2,0))    
    titles = ['Low Exposure Weight', 'Mid Exposure Weight', 'High Exposure Weight']
    for i in range(3):
        plt.figure(figsize=(6,6), dpi=dpi)
        # Normalize weights to [0,1] range for better visualization
        weights = weight_vis_np[:,:,i]
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        plt.imshow(weights, cmap='viridis', vmin=0, vmax=1)
        plt.title(titles[i])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{epoch}_{index}_weight_{i}.png', dpi=dpi)
        plt.close()
            
            
# 使用双线性插值调整张量大小
def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

# 根据光流场对图像进行变形
def warp(img, flow):
    B, _, H, W = flow.shape
    # 生成归一化的网格坐标 [-1,1]
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)  # x方向坐标
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)  # y方向坐标
    grid = torch.cat([xx, yy], 1).to(img)  # 合并x,y坐标得到采样网格
    # 将光流场归一化到[-1,1]范围
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    # 将网格和光流相加得到采样位置,并调整维度顺序
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    # 使用grid_sample进行双线性插值采样
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

# 计算三重曝光图像中低曝光权重
def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w

# 计算三重曝光图像中中等曝光权重
def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w

# 计算三重曝光图像中高曝光权重
def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w

    # Visualize weights with different colors
def visualize_weights(w_low, w_mid, w_high):
    # Create separate heatmaps for each weight
    w_low_vis = w_low[:,0:1]  # Low exposure weights heatmap
    w_mid_vis = w_mid[:,0:1]  # Mid exposure weights heatmap 
    w_high_vis = w_high[:,0:1]  # High exposure weights heatmap
    
    # Stack them into a single tensor along channel dimension
    vis = torch.cat([w_low_vis, w_mid_vis, w_high_vis], dim=1)
    return vis


def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
    sum_img = torch.zeros_like(ldr_imgs[1])
    sum_w = torch.zeros_like(ldr_imgs[1])
    w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
    w_mid = weight_3expo_mid_tog17(ldr_imgs[1]) + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0) + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
    w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
    w_list = [w_low, w_mid, w_high]
    for i in range(len(ldr_imgs)):
        sum_w += w_list[i]
        sum_img += w_list[i] * lin_imgs[i]
    hdr_img = sum_img / (sum_w + 1e-9)
    return hdr_img
# 对HDR图像进行范围压缩
def range_compressor(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)

# 计算两张图像之间的PSNR值
def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr

def tone_mapping(hdr_img):
    """
    Reinhard tone mapping for HDR images
    Reference: Reinhard et al., "Photographic Tone Reproduction for Digital Images"
    
    Args:
        hdr_img: Input HDR image in HWC format
    Returns:
        Tone mapped LDR image
    """
    # Convert to luminance
    luminance = 0.2126 * hdr_img[..., 0] + 0.7152 * hdr_img[..., 1] + 0.0722 * hdr_img[..., 2]
    # Calculate average luminance (using log average)
    L_avg = np.exp(np.mean(np.log(luminance + 1e-6)))
    # Scale luminance
    L_scaled = luminance * 0.18 / L_avg
    # Apply tone mapping
    L_mapped = L_scaled / (1 + L_scaled)
    # Preserve colors
    ratio = L_mapped / (luminance + 1e-6)
    ldr_img = np.zeros_like(hdr_img)
    for i in range(3):
        ldr_img[..., i] = hdr_img[..., i] * ratio
    return np.clip(ldr_img, 0, 1)

def visualize_results(img0_c, img1_c, img2_c, img_hdr_m, img_hdr_gt, masks, epoch, index,output_dir='output', dpi=450):
    """
    Visualize HDR reconstruction results including input images, flows, masks and predictions
    
    Args:
        img0_c, img1_c, img2_c: Input image tensors (concatenated LDR and linear)
        img_hdr_m: Predicted HDR image tensor
        img_hdr_gt: Ground truth HDR image tensor  
        masks: List of [flow0, flow2, mask0, mask2] tensors
        epoch: Current epoch number
        index: Image index in current validation
        output_dir: Directory to save visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Extract flows and masks
    flow0, flow2, mask0, weight_vis, mask2 = masks
    
    # Convert tensors to numpy (take first batch)
    img0 = img0_c[0, 3:6].cpu().numpy()  # Take LDR part
    img1 = img1_c[0, 3:6].cpu().numpy()
    img2 = img2_c[0, 3:6].cpu().numpy()
    pred_hdr = img_hdr_m[0].cpu().numpy()
    gt_hdr = img_hdr_gt[0].cpu().numpy()
    
    # Calculate flow magnitudes
    flow0_mag = torch.sqrt(flow0[0,0]**2 + flow0[0,1]**2).cpu().numpy()
    flow2_mag = torch.sqrt(flow2[0,0]**2 + flow2[0,1]**2).cpu().numpy()
    mask0_vis = mask0[0,0].cpu().numpy()
    mask2_vis = mask2[0,0].cpu().numpy()

    # Transpose all images from CHW to HWC
    img0 = np.transpose(img0, (1,2,0))
    img1 = np.transpose(img1, (1,2,0))
    img2 = np.transpose(img2, (1,2,0))
    pred_hdr = np.transpose(pred_hdr, (1,2,0))
    gt_hdr = np.transpose(gt_hdr, (1,2,0))

    # Apply tone mapping
    pred_hdr_tm = tone_mapping(pred_hdr)
    gt_hdr_tm = tone_mapping(gt_hdr)

    # Create visualization plot
    plt.figure(dpi=dpi)  # Set higher DPI
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    # First row - input images
    axs[0,0].imshow(img0)
    axs[0,0].set_title('Input 0')
    axs[0,1].imshow(img1)
    axs[0,1].set_title('Input 1') 
    axs[0,2].imshow(img2)
    axs[0,2].set_title('Input 2')

    # Second row - flows and predicted HDR
    im1 = axs[1,0].imshow(flow0_mag, cmap='hot')
    axs[1,0].set_title('Flow 0 Magnitude')
    plt.colorbar(im1, ax=axs[1,0])
    
    axs[1,1].imshow(pred_hdr_tm)
    axs[1,1].set_title('Predicted HDR (tone mapped)')
    
    im2 = axs[1,2].imshow(flow2_mag, cmap='hot')
    axs[1,2].set_title('Flow 2 Magnitude')
    plt.colorbar(im2, ax=axs[1,2])

    # Third row - masks and ground truth HDR
    # Use 'rainbow' colormap for more vibrant visualization
    im3 = axs[2,0].imshow(mask0_vis, cmap='rainbow', vmin=0, vmax=1)
    axs[2,0].set_title('Mask 0')
    plt.colorbar(im3, ax=axs[2,0])
    
    axs[2,1].imshow(gt_hdr_tm)
    axs[2,1].set_title('Ground Truth HDR (tone mapped)')
    
    # Use 'rainbow' colormap for more vibrant visualization
    im4 = axs[2,2].imshow(mask2_vis, cmap='rainbow', vmin=0, vmax=1)
    axs[2,2].set_title('Mask 2')
    plt.colorbar(im4, ax=axs[2,2])

    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with high DPI and close
    plt.savefig(f'{output_dir}/{epoch}_{index}.png', dpi=dpi)
    wandb.log(
        {
            "epoch": epoch,
            "index": index,
            "plot": wandb.Image(plt)
        }
    )
    plt.close()
    # Save weight visualization separately for each exposure
    # visualize_weights(weight_vis, output_dir, epoch, index, dpi)

def create_similarity_mask(img1, img2, window_size=7, sigma=0.8, c1=0.01, c2=0.03, mu_scale=1):
    """
    Create a similarity mask between two images that is robust to illumination changes
    Args:
        img1, img2: Input images (NCHW format)
        window_size: Size of the gaussian window
        sigma: Standard deviation for gaussian window
        c1, c2: Constants for numerical stability
    Returns:
        Similarity mask where high values indicate similar regions
    """
    # Convert to grayscale using luminance weights if in color
    img1 = img1[:, 3:6]
    img2 = img2[:, 3:6]
    if img1.shape[1] == 3:
        img1_gray = 0.2126 * img1[:,0:1] + 0.7152 * img1[:,1:2] + 0.0722 * img1[:,2:3]
        img2_gray = 0.2126 * img2[:,0:1] + 0.7152 * img2[:,1:2] + 0.0722 * img2[:,2:3]
    else:
        img1_gray = img1
        img2_gray = img2

    # Create gaussian window
    gaussian_window = torch.Tensor([
        [np.exp(-(x**2 + y**2)/(2*sigma**2)) 
         for x in range(-(window_size//2), window_size//2 + 1)]
        for y in range(-(window_size//2), window_size//2 + 1)]
    ).to(img1.device)
    
    gaussian_window = gaussian_window / gaussian_window.sum()
    gaussian_window = gaussian_window.view(1, 1, window_size, window_size)

    # Compute local means and standard deviations
    pad_size = window_size // 2
    padded1 = F.pad(img1_gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    padded2 = F.pad(img2_gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    mu1 = F.conv2d(padded1, gaussian_window, groups=1)
    mu2 = F.conv2d(padded2, gaussian_window, groups=1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    mu12*= mu_scale
    
    sigma1_sq = F.conv2d(padded1 * padded1, gaussian_window, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(padded2 * padded2, gaussian_window, groups=1) - mu2_sq
    sigma12 = F.conv2d(padded1 * padded2, gaussian_window, groups=1) - mu12
    
    # Compute structural similarity
    C1 = (c1 * torch.max(img1_gray).item()) ** 2
    C2 = (c2 * torch.max(img1_gray).item()) ** 2
    
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Normalize to [0,1]
    ssim_map = (ssim_map + 1) / 2
    
    return ssim_map



# Global RAFT model
model = None

def create_flow_similarity_mask(img1, img2, max_displacement=12.0):
    """
    Create a similarity mask based on motion between two images
    """
    global model
    if model is None:
        model = raft_large(pretrained=True).to(img1.device)
        model.eval()

    with torch.no_grad():
        # 使用LDR部分
        img1_norm = img1[:, 3:6]
        img2_norm = img2[:, 3:6]
        
        # 获取原始尺寸
        _, _, H, W = img1_norm.shape
        
        # 计算需要的填充
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        # 添加填充
        if pad_h > 0 or pad_w > 0:
            img1_norm = F.pad(img1_norm, (0, pad_w, 0, pad_h))
            img2_norm = F.pad(img2_norm, (0, pad_w, 0, pad_h))
        
        # 计算光流
        flow = model(img1_norm, img2_norm)[-1]
        
        # 如果有填充，去除填充部分
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H, :W]
        
        # 计算光流幅度
        flow_magnitude = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2)
        
        # 转换为相似度mask
        similarity = torch.exp(-flow_magnitude / max_displacement)
        # Find threshold at 80th percentile
        threshold = torch.quantile(similarity, 0.1)
        # Convert to binary mask based on threshold
        similarity = (similarity >= threshold).float()
     
        return similarity,flow

def exposure_aware_hdr(ldr_img, exposure_value, gamma=2.2, window_size=15):
    """
    Convert LDR to HDR with known exposure value
    Args:
        ldr_img: Input LDR image [N,C,H,W] in range [0,1]
        exposure_value: EV value (e.g., 2.0 means 2 stops overexposed)
        gamma: Camera gamma
        window_size: Local window size
    Returns:
        HDR image
    """
    # 1. 计算曝光比例 (2^EV)
    exposure_scale = 2.0 ** exposure_value
    
    # 2. 反gamma校正得到线性值
    linear = ldr_img ** gamma
    
    # 3. 补偿曝光
    linear_compensated = linear / exposure_scale
    
    # 4. 计算亮度
    luminance = 0.2126 * linear_compensated[:,0:1] + \
                0.7152 * linear_compensated[:,1:2] + \
                0.0722 * linear_compensated[:,2:3]
    
    # 5. 局部色调映射
    # 创建高斯窗口
    sigma = window_size / 6
    gaussian = torch.Tensor([
        [np.exp(-(x**2 + y**2)/(2*sigma**2)) 
         for x in range(-(window_size//2), window_size//2 + 1)]
        for y in range(-(window_size//2), window_size//2 + 1)]
    ).to(ldr_img.device)
    gaussian = gaussian / gaussian.sum()
    gaussian = gaussian.view(1, 1, window_size, window_size)
    
    # 计算局部平均亮度
    pad_size = window_size // 2
    padded = F.pad(luminance, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    local_mean = F.conv2d(padded, gaussian)
    
    # 分离细节层
    detail_layer = luminance / (local_mean + 1e-6)
    
    # 6. 动态范围扩展 (考虑曝光补偿后的值)
    base_expand = torch.where(
        local_mean > 0.5,
        1.0 + (local_mean - 0.5) * 2.0,  # 高亮区域
        1.0 + local_mean * 0.5           # 暗区域
    )
    
    # 7. 重建HDR
    expanded_base = local_mean * base_expand
    hdr_luminance = expanded_base * detail_layer
    
    # 8. 恢复颜色
    hdr = torch.zeros_like(linear_compensated)
    for c in range(3):
        hdr[:,c:c+1] = linear_compensated[:,c:c+1] * (hdr_luminance / (luminance + 1e-6))
    
    return hdr

def advanced_single_image_hdr(ldr_img, gamma=2.2, window_size=15):
    """
    More advanced single image HDR conversion with local adaptation
    """
    return exposure_aware_hdr(ldr_img=ldr_img, exposure_value=2.0)