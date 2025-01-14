import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import warp, merge_hdr,create_flow_similarity_mask


div_size = 16
div_flow = 20.0


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(6, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        
    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(122, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 2)

    def forward(self, f0, f1, f2, mask0, mask2):
        f_in = torch.cat([f0,  f1, f2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 0:1]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 1:2]
        return up_mask0, up_mask2


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv0 = nn.Sequential(convrelu(6, 30), convrelu(30, 40))
        self.conv1 = nn.Sequential(convrelu(6, 50), convrelu(50, 60))
        self.conv2 = nn.Sequential(convrelu(6, 30), convrelu(30, 40))
        self.resblock1 = ResBlock(140, 1)
        self.resblock2 = ResBlock(140, 2)
        self.resblock3 = ResBlock(140, 4)
        self.resblock4 = ResBlock(140, 2)
        self.resblock5 = ResBlock(140, 1)
        self.conv3 = nn.Conv2d(140, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
      


    def forward(self, img0_c, img1_c, img2_c):
        img = img1_c[:,3:6,:,:]
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(img1_c)
        feat2 = self.conv2(img2_c)

        feat = torch.cat([feat0, feat1, feat2], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        res = self.conv3(feat)
 
 
        img_hdr_r = torch.sigmoid(img + res)
        return img_hdr_r  

class MaskRefineNet(nn.Module):
    def __init__(self):
        super(MaskRefineNet, self).__init__()
        self.conv0 = nn.Sequential(convrelu(6, 30), convrelu(30, 40))
        self.conv1 = nn.Sequential(convrelu(9, 50), convrelu(50, 60))
        self.conv2 = nn.Sequential(convrelu(6, 30), convrelu(30, 40))
        self.resblock1 = ResBlock(140, 1)
        self.resblock2 = ResBlock(140, 2)
        self.resblock3 = ResBlock(140, 4)
        self.resblock4 = ResBlock(140, 2)
        self.resblock5 = ResBlock(140, 1)
        self.conv3 = nn.Conv2d(140, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
      
    def forward(self, img0_c, img1_c, img2_c,img_hdr_r):
        img = img1_c[:,3:6,:,:]
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([img1_c,img_hdr_r],dim=1))
        feat2 = self.conv2(img2_c)

        feat = torch.cat([feat0, feat1, feat2], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        feat = self.conv3(feat)
        scale_factor = 20.0 
        mask = torch.sigmoid(scale_factor * feat)
        return mask

class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()
        self.mask_refinenet = MaskRefineNet()
        self.refinenet = RefineNet()
         
    def forward(self, img0_c, img1_c, img2_c):
        img_hdr_r    = self.refinenet(img0_c, img1_c, img2_c)
        mask = self.mask_refinenet(img0_c, img1_c, img2_c,img_hdr_r)

        return img_hdr_r ,mask
    
