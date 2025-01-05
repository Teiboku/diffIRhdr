import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp, merge_hdr,create_flow_similarity_mask


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
        self.conv1 = convrelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


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

class MyRefineNet(nn.Module):
    def __init__(self):
        super(MyRefineNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            convrelu(20, 64),
            convrelu(64, 64),
            ResBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            convrelu(64, 128),
            convrelu(128, 128),
            ResBlock(128)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), 
            convrelu(128, 256),
            convrelu(256, 256),
            ResBlock(256)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            convrelu(256, 512),
            convrelu(512, 512),
            ResBlock(512)
        )
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            convrelu(512, 1024),
            convrelu(1024, 1024),
            ResBlock(1024),
            deconv(1024, 512)
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            convrelu(1024, 512),
            convrelu(512, 512),
            ResBlock(512),
            deconv(512, 256)
        )
        self.dec3 = nn.Sequential(
            convrelu(512, 256),
            convrelu(256, 256),
            ResBlock(256),
            deconv(256, 128)
        )
        self.dec2 = nn.Sequential(
            convrelu(256, 128),
            convrelu(128, 128),
            ResBlock(128),
            deconv(128, 64)
        )
        self.dec1 = nn.Sequential(
            convrelu(128, 64),
            convrelu(64, 64),
            ResBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, img0_c, img1_c, img2_c, mask0, mask2):
        # Check and pad input dimensions to be multiple of 16
        h, w = img0_c.shape[2:]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            # Pad all inputs
            img0_c = F.pad(img0_c, (0, pad_w, 0, pad_h))
            img1_c = F.pad(img1_c, (0, pad_w, 0, pad_h))
            img2_c = F.pad(img2_c, (0, pad_w, 0, pad_h))
            mask0 = F.pad(mask0, (0, pad_w, 0, pad_h))
            mask2 = F.pad(mask2, (0, pad_w, 0, pad_h))

        # Concatenate input images with masks
        x = torch.cat([img0_c, img1_c, img2_c, mask0, mask2], 1)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bridge
        bridge = self.bridge(enc4)
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([bridge, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        
        # Residual connection
        img_hdr_r = torch.clamp(img1_c[:,3:6,:,:] + dec1, 0, 1)
        
        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            img_hdr_r = img_hdr_r[:,:,:h,:w]
            
        return img_hdr_r

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv0 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))
        self.conv1 = nn.Sequential(convrelu(6+2+2+1+1+3, 40), convrelu(40, 40))
        self.conv2 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))
        self.resblock1 = ResBlock(80, 1)
        self.resblock2 = ResBlock(80, 2)
        self.resblock3 = ResBlock(80, 4)
        self.resblock4 = ResBlock(80, 2)
        self.resblock5 = ResBlock(80, 1)
        self.conv3 = nn.Conv2d(80, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        
    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([img1_c, flow0 / div_flow, flow2 / div_flow, mask0, mask2, img_hdr_m], 1))
        feat2 = self.conv2(img2_c)
        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        res = self.conv3(feat)
        img_hdr_r = torch.clamp(img_hdr_m + res, 0, 1)
        return img_hdr_r


class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (int(div_size * np.ceil(h * scale_factor / div_size)), int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5)
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4)
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3)
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(up_mask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_mask2_1 = F.interpolate(up_mask2_1, size=org_size, mode='bilinear', align_corners=False)

        up_mask0_1 = torch.sigmoid(up_mask0_1)
        up_mask2_1 = torch.sigmoid(up_mask2_1)

        return up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1
    
    def forward(self, img0_c, img1_c, img2_c, scale_factor=0.5, refine=True):
        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        # Get flows and masks from forward_flow_mask
        
        flow0, flow2, mask0, mask2 = self.forward_flow_mask(img0_c, img1_c, img2_c, scale_factor=scale_factor)
        img0_c_warp = warp(img0_c, flow0)
        img2_c_warp = warp(img2_c, flow2)
        img_hdr_m = merge_hdr(
            [img0_c_warp[:, 3:6, :, :], img1_c[:, 3:6, :, :], img2_c_warp[:, 3:6, :, :]], 
            [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
            mask0, mask2
        )
        if refine == True:
            img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m)
            return img_hdr_r
        #[flow0, flow2, mask0, weight_vis,mask2]
 

        return img_hdr_m
    #[flow0, flow2, mask0, weight_vis,mask2]