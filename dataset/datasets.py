import os
from glob import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

#from utils.mask_util.hdr_mask import get_diff_mask

def read_ldr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1])
    img = (img / 2 ** 16).clip(0, 1).astype(np.float32)
    return img

def read_hdr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1]).astype(np.float32)
    return img

def read_expos(txt_path):
    expos = np.power(2, np.loadtxt(txt_path) - min(np.loadtxt(txt_path))).astype(np.float32)
    return expos


class HDRDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, patch_size=None):
        """
        Args:
            root_dir (str): Dataset root directory path
            transform: Data preprocessing transforms
            is_train (bool): Whether this is training set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        if is_train:
            self.patch_size = patch_size
        else:
            self.patch_size = None
        # Set data subdirectory
        self.data_dir = os.path.join(root_dir, 'Training' if is_train else 'Test')
        
        sequences = sorted(os.listdir(self.data_dir))
        print(len(sequences))
        
        self.img0_list = []
        self.img1_list = []
        self.img2_list = [] 
        self.gt_list = []
        self.expos_list = []
        
        for seq in sequences:
            seq_dir = os.path.join(self.data_dir, seq)
            ldr_list = sorted(glob(os.path.join(seq_dir, '*.tif')))
            assert len(ldr_list) == 3
            self.img0_list.append(ldr_list[0])
            self.img1_list.append(ldr_list[1]) 
            self.img2_list.append(ldr_list[2])
            
            hdr_list = sorted(glob(os.path.join(seq_dir, '*.hdr')))
            assert len(hdr_list) == 1
            self.gt_list.append(hdr_list[0])
            
            expo_list = sorted(glob(os.path.join(seq_dir, '*.txt')))
            assert len(expo_list) == 1
            self.expos_list.append(expo_list[0])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
        """
        # Convert images to CUDA tensors
        img0_tensor = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).cuda()
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).cuda()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).cuda()
        
        diff_mask_0 = get_diff_mask(img0_tensor, img1_tensor)
        diff_mask_1 = get_diff_mask(img2_tensor, img1_tensor)
        
        # Save masks to same directory as input images
        mask_dir = os.path.dirname(self.img0_list[idx])
        mask0_path = os.path.join(mask_dir, f'mask0.png')
        mask1_path = os.path.join(mask_dir, f'mask1.png')
        
        cv2.imwrite(mask0_path, (diff_mask_0.cpu().numpy()[0,0] * 255).astype(np.uint8))
        cv2.imwrite(mask1_path, (diff_mask_1.cpu().numpy()[0,0] * 255).astype(np.uint8))
        """
        gt = read_hdr(self.gt_list[idx])
        expos = read_expos(self.expos_list[idx])
        imgs_ldr = [img0.copy(), img1.copy(), img2.copy()]
        imgs_lin = []
        for i in range(3):
            img_lin = (imgs_ldr[i] ** 2.2) / expos[i]
            img_lin = np.transpose(img_lin, (2, 0, 1))
            imgs_lin.append(img_lin)
            imgs_ldr[i] = np.transpose(imgs_ldr[i], (2, 0, 1))
        gt = np.transpose(gt.copy(), (2, 0, 1))

        if self.patch_size is not None:
            h, w = imgs_lin[0].shape[1:]
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            for i in range(3):
                imgs_lin[i] = imgs_lin[i][:, top:top+self.patch_size, left:left+self.patch_size]
                imgs_ldr[i] = imgs_ldr[i][:, top:top+self.patch_size, left:left+self.patch_size]
            gt = gt[:, top:top+self.patch_size, left:left+self.patch_size]
  
  
        # Random channel order reversal augmentation
        if self.is_train and random.random() < 0.3:
            # Reverse RGB channel order for all images
            for i in range(3):
                imgs_lin[i] = imgs_lin[i][::-1, :, :].copy()  # 添加 .copy()
                imgs_ldr[i] = imgs_ldr[i][::-1, :, :].copy()  # 添加 .copy()
            gt = gt[::-1, :, :].copy()  # 添加 .copy()
        return imgs_lin, imgs_ldr, expos, gt

         