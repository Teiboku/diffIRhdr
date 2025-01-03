import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import HDRDataset,SIGGRAPH17_Test_Dataset
from models.SAFNet import SAFNet
from utils import range_compressor, calculate_psnr, visualize_results,advanced_single_image_hdr,prepare_input_images
from loss import census_loss
import matplotlib.pyplot as plt
import wandb

def init_wandb():
    wandb.init(
        project="diffirhdr",
        config={
        }
)

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(train_loader):
        img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, device)
        img_hdr_m, new_mask = model(img0_c, img1_c, img2_c)
        img_hdr_gt = img_hdr_gt.to(device)
        census = census_loss(img_hdr_m, img_hdr_gt)
        img_hdr_r_m = range_compressor(img_hdr_m)
        img_hdr_gt_m = range_compressor(img_hdr_gt)
        compressed_loss = torch.nn.MSELoss()(img_hdr_r_m, img_hdr_gt_m)
        loss =  0.3*census +  compressed_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            
def validate_and_save(model, val_loader, device, epoch):
    model.eval()
    val_psnr = []
    with torch.no_grad():
        for i, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(val_loader):
            img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, device)
            img_hdr_gt = img_hdr_gt.to(device)        
            img_hdr_m, masks = model(img0_c, img1_c, img2_c)
            psnr = calculate_psnr(img_hdr_m.to('cuda'), img_hdr_gt)
            val_psnr.append(psnr.item())
            visualize_results(img0_c, img1_c, img2_c, img_hdr_m, img_hdr_gt, masks, epoch, i, dpi=300)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        print(f'Validation PSNR at epoch {epoch}: {avg_psnr:.2f}')
        wandb.log({"psnr": avg_psnr})
    torch.save(model.state_dict(), f'checkpoints/safnet_epoch_{epoch}.pth')
    return avg_psnr



def train_safnet(
    train_dir='/root/code/hdr/SIGGRAPH17_HDR_Trainingset',
    val_dir='/root/code/hdr/SIGGRAPH17_HDR_Testset/Test',
    num_epochs=200,
    batch_size=1,
    learning_rate=1e-4,
    device='cuda'
):
    model = SAFNet().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_dataset = HDRDataset(train_dir, is_train=True)
    val_dataset = SIGGRAPH17_Test_Dataset(val_dir)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    for epoch in range(num_epochs):    
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        if epoch % 10 == 0:
            validate_and_save(model, val_loader, device, epoch)
if __name__ == '__main__':
    init_wandb()
    train_safnet()
