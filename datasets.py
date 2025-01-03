import os
from glob import glob
import numpy as np
import cv2
from torch.utils.data import Dataset


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


class SIGGRAPH17_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        sequences = sorted(os.listdir(dataset_dir))
        print(len(sequences))
        self.img0_list = []
        self.img1_list = []
        self.img2_list = []
        self.gt_list = []
        self.expos_list = []
        for seq in sequences:
            ldr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.tif')))
            assert len(ldr_list) == 3
            self.img0_list.append(ldr_list[0])
            self.img1_list.append(ldr_list[1])
            self.img2_list.append(ldr_list[2])
            hdr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.hdr')))
            assert len(hdr_list) == 1
            self.gt_list.append(hdr_list[0])
            expo_list = sorted(glob(os.path.join(dataset_dir, seq, '*.txt')))
            assert len(expo_list) == 1
            self.expos_list.append(expo_list[0])

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
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
            
        return imgs_lin, imgs_ldr, expos, gt
 

class HDRDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (str): Dataset root directory path
            transform: Data preprocessing transforms
            is_train (bool): Whether this is training set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
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
            
        return imgs_lin, imgs_ldr, expos, gt