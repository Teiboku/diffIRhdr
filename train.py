import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset.datasets import HDRDataset
from dataset.transform import HDRTransform
from models.SAFNet import MyRefineNet, SAFNet
from utils import prepare_masks, range_compressor, calculate_psnr, tone_mapping, visualize_results,advanced_single_image_hdr,prepare_input_images
from loss import census_loss
import matplotlib.pyplot as plt
import wandb
from models.archs.NAFNet_arch import NAFNet

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
        mask0 , mask2 = prepare_masks(imgs_ldr,device)
        #img0_c = img0_c * (1-mask0)
        #img2_c = img2_c * (1-mask2)
        img_hdr_m = model(img0_c, img1_c, img2_c, mask0 , mask2)
        img_hdr_gt = img_hdr_gt.to(device)
        #census = census_loss(img_hdr_m, img_hdr_gt)
        img_hdr_r_m = range_compressor(img_hdr_m)
        img_hdr_gt_m = range_compressor(img_hdr_gt)
        compressed_loss = torch.nn.L1Loss()(img_hdr_r_m, img_hdr_gt_m)
        loss =  compressed_loss
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
            mask0 , mask2 = prepare_masks(imgs_ldr,device)

            img_hdr_gt = img_hdr_gt.to(device)        
            img_hdr_m = model(img0_c, img1_c, img2_c, mask0 , mask2)
            psnr = calculate_psnr(img_hdr_m.to('cuda'), img_hdr_gt)
            val_psnr.append(psnr.item())
            # Log HDR images to wandb
            if i == 0:  # Only log first batch to avoid too many images
                wandb.log({
                    f"epoch_{epoch}_predicted": wandb.Image(tone_mapping(img_hdr_m[0].cpu().numpy().transpose(1,2,0))),
                    f"epoch_{epoch}_ground_truth": wandb.Image(tone_mapping(img_hdr_gt[0].cpu().numpy().transpose(1,2,0)))
                })

            # visualize_results(img0_c, img1_c, img2_c, img_hdr_m, img_hdr_gt, mask0, mask2, epoch, i, dpi=300)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        print(f'Validation PSNR at epoch {epoch}: {avg_psnr:.2f}')
        wandb.log({"psnr": avg_psnr})
    torch.save(model.state_dict(), f'checkpoints/safnet_epoch_{epoch}.pth')
    return avg_psnr



def train_safnet(
    train_dir='/root/code/hdr/kata17',
    val_dir='/root/code/hdr/kata17',
    num_epochs=200,
    batch_size=2,
    learning_rate=1e-4,
    device='cuda',
    ckpt_path='./checkpoints/safnet_epoch_20.pth'
):
    model = NAFNet(width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1]).to(device)
    #model.load_state_dict(torch.load(ckpt_path))
   
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_dataset = HDRDataset(train_dir,transform=HDRTransform(patch_size=512), is_train=True)
    val_dataset = HDRDataset(val_dir,is_train=False)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    for epoch in range(num_epochs):    
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        if epoch % 20 == 0 and epoch != 0:
            validate_and_save(model, val_loader, device, epoch)




if __name__ == '__main__':
    init_wandb()
    train_safnet()
