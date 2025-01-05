import torch
import torch.nn.functional as F
def census_transform(img, window_size=7):
    """
    Census 变换
    Args:
        img: [B,C,H,W] 图像
    """
    B, C, H, W = img.shape
    pad_size = window_size // 2
    
    # 对图像进行填充
    padded = F.pad(img, [pad_size]*4, mode='reflect')
    
    # 获取中心像素值
    center_pixels = img
    
    # 存储二进制特征 - 为每个通道创建独立的census特征
    census = torch.zeros((B, C*(window_size*window_size-1), H, W), 
                        device=img.device, dtype=torch.float32)
    
    # 计算Census变换
    idx = 0
    for i in range(window_size):
        for j in range(window_size):
            if i == pad_size and j == pad_size:
                continue 
            neighbor = padded[:, :, i:i+H, j:j+W]
            census[:, idx*C:(idx+1)*C] = (neighbor < center_pixels).float()
            idx += 1
    return census

def census_loss(pred, target, window_size=7):
    """
    Census Loss计算
    Args:
        pred: 预测图像 [B,1,H,W]
        target: 目标图像 [B,1,H,W]
        window_size: Census变换的窗口大小
    """
    # 计算Census变换
    pred_census = census_transform(pred, window_size)
    target_census = census_transform(target, window_size)
    
    # 计算Hamming距离
    loss = torch.abs(pred_census - target_census).mean()
    
    return loss