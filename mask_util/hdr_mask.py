import torch
import torch.nn.functional as F
import kornia.color as K


def rgb_to_lab(rgb_img):
    return K.rgb_to_lab(rgb_img)

def lab_to_rgb(lab_img):
    return K.lab_to_rgb(lab_img)

def histogram_matching(source, reference):
    B, C, H, W = source.shape
    matched = torch.zeros_like(source)
    
    for b in range(B):
        for c in range(C):
            src = source[b,c].flatten()
            ref = reference[b,c].flatten()
            
            src_sorted, src_indices = torch.sort(src)
            ref_sorted, _ = torch.sort(ref)
            
            mapping = torch.zeros_like(src)
            mapping[src_indices] = ref_sorted
            
            matched[b,c] = mapping.reshape(H, W)
            
    return matched

def match_brightness(source, reference):
    source = torch.clamp(source, 0, 1)
    reference = torch.clamp(reference, 0, 1)
    
    source_lab = rgb_to_lab(source)
    reference_lab = rgb_to_lab(reference)
    
    source_l = source_lab[:, 0:1, :, :]
    reference_l = reference_lab[:, 0:1, :, :]
    
    matched_l = histogram_matching(source_l, reference_l)
    
    matched_lab = torch.cat([matched_l, source_lab[:, 1:, :, :]], dim=1)
    
    matched_rgb = lab_to_rgb(matched_lab)
    
    matched_rgb = torch.clamp(matched_rgb, 0, 1)
    
    return matched_rgb

def average_filter(x, kernel_size=3):
    """
    Apply average filter while maintaining input shape
    Args:
        x: Input tensor of shape (B, C, H, W)
        kernel_size: Size of the averaging kernel
    Returns:
        Filtered tensor of shape (B, C, H, W)
    """
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
        
    kernel = torch.ones(x.shape[1], 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    kernel = kernel.to(x.device)
    
    # 计算padding
    padding = kernel_size // 2
    
    # 使用conv2d进行平均滤波
    out = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    
    # 确保输出shape与输入完全相同
    if out.shape != x.shape:
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
    return out

def morphological_opening(mask, eroded_kernel_size=11, dilated_kernel_size=11):
    """
    Apply morphological opening while maintaining input shape
    Args:
        mask: Input tensor of shape (B, 1, H, W)
        eroded_kernel_size: Size of erosion kernel (will be forced to odd)
        dilated_kernel_size: Size of dilation kernel (will be forced to odd)
    Returns:
        Opened mask of same shape as input
    """
    # 确保kernel sizes是奇数
    if eroded_kernel_size % 2 == 0:
        eroded_kernel_size += 1
    if dilated_kernel_size % 2 == 0:
        dilated_kernel_size += 1
        
    # 记住原始shape
    original_shape = mask.shape
    
    # Erosion
    eroded = -(F.max_pool2d(-mask, 
                           kernel_size=eroded_kernel_size, 
                           stride=1, 
                           padding=eroded_kernel_size//2))
    
    # Dilation
    opened = F.max_pool2d(eroded, 
                         kernel_size=dilated_kernel_size,
                         stride=1, 
                         padding=dilated_kernel_size//2)
    
    # 确保输出shape与输入相同
    if opened.shape != original_shape:
        opened = F.interpolate(opened, 
                             size=(original_shape[2], original_shape[3]),
                             mode='nearest')
    
    return opened

def detect_exposure_regions(img_tensor, percentile=0.05):
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
 
    if img_tensor.shape[1] == 3:  
        luminance = 0.299 * img_tensor[:,0] + 0.587 * img_tensor[:,1] + 0.114 * img_tensor[:,2]
    else:  
        luminance = img_tensor[:,0]
        
    # 计算最暗和最亮的阈值(前后5%)
    under_threshold = torch.quantile(luminance.flatten(), percentile)
    over_threshold = torch.quantile(luminance.flatten(), 1 - percentile)
    
    # 生成mask
    under_exposed = (luminance <= under_threshold).float()
    over_exposed = (luminance >= over_threshold).float()
    
    return under_exposed, over_exposed

def get_diff_mask(source, reference,kernel_size=17,threshold=0.93,eroded_kernel_size=11,dilated_kernel_size=13,percentile=0.03):
    source = average_filter(source,kernel_size=kernel_size)
    reference = average_filter(reference,kernel_size=kernel_size)
    
    source = match_brightness(source, reference)
    under_exposed, over_exposed = detect_exposure_regions(source,percentile=percentile)
    diff = torch.abs(source - reference)
    
    diff_mean = diff.mean(dim=1, keepdim=True) # [B,1,H,W]
    diff_mean = diff_mean * (1-under_exposed.unsqueeze(1)) * (1-over_exposed.unsqueeze(1))
    # Find threshold for top percentage
    threshold = torch.quantile(diff_mean.flatten(), threshold)
    
    # Create binary mask for differences above threshold
    mask = (diff_mean > threshold).float()
    mask = morphological_opening(mask, eroded_kernel_size=eroded_kernel_size, dilated_kernel_size=dilated_kernel_size)
    return mask[:,0:1] # Only return single channel [B,1,H,W]