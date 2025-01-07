# train.py
import torch
import wandb
from utils.loss import perceptual_loss
from utils.utils import calculate_psnr, prepare_input_images, prepare_masks, range_compressor, tone_mapping
from utils.config import Config
from utils.train_utils import init_wandb, get_model, get_optimizer, get_scheduler, get_dataloader

def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch, config):
    model.train()
    for batch_idx, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(train_loader):
        img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, device)
        

        if config.model['name'] == "SAFNet":
            img_hdr_m = model(img0_c, img1_c, img2_c, refine=False)
        else:
            img_hdr_m = model(torch.cat([img0_c, img1_c, img2_c], dim=1))
        img_hdr_gt = img_hdr_gt.to(device)
        #census = census_loss(img_hdr_m, img_hdr_gt)
        img_hdr_r_m = range_compressor(img_hdr_m)
        img_hdr_gt_m = range_compressor(img_hdr_gt)
        compressed_loss = torch.nn.L1Loss()(img_hdr_r_m, img_hdr_gt_m)  
        p_loss = perceptual_loss(img_hdr_m, img_hdr_gt)
        loss =  compressed_loss + 0.1 * p_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if batch_idx % 10 == 0:
            wandb.log({"loss": loss.item()})

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            
def validate_and_save(model, val_loader, device, epoch, config):
    model.eval()
    val_psnr = []
    model_name = config.model['name']
    with torch.no_grad():
        for i, (imgs_lin, imgs_ldr, expos, img_hdr_gt) in enumerate(val_loader):
            img0_c, img1_c, img2_c = prepare_input_images(imgs_lin, imgs_ldr, device)
            img_hdr_gt = img_hdr_gt.to(device)   
            if model_name == "SAFNet":
                img_hdr_m = model(img0_c, img1_c, img2_c, refine=False)
            else:
                img_hdr_m = model(torch.cat([img0_c, img1_c, img2_c], dim=1))
            psnr = calculate_psnr(img_hdr_m.to('cuda'), img_hdr_gt)
            val_psnr.append(psnr.item())
            # Log HDR images to wandb
            if i % 2 == 0:  # Only log first batch to avoid too many images
                wandb.log({
                    "epoch": epoch,
                    "batch": i,
                    "predicted_image": wandb.Image(
                        tone_mapping(img_hdr_m[0].cpu().numpy().transpose(1, 2, 0)),
                        caption=f"Epoch {epoch}, Batch {i} - Predicted HDR Image"
                    ),
                    "ground_truth_image": wandb.Image(
                        tone_mapping(img_hdr_gt[0].cpu().numpy().transpose(1, 2, 0)),
                        caption=f"Epoch {epoch}, Batch {i} - Ground Truth HDR Image"
                    )
                })
            # visualize_results(img0_c, img1_c, img2_c, img_hdr_m, img_hdr_gt, mask0, mask2, epoch, i, dpi=300)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        print(f'Validation PSNR at epoch {epoch}: {avg_psnr:.2f}')
        wandb.log({"psnr": avg_psnr})
    torch.save(model.state_dict(), f'checkpoints/{model_name}_epoch_{epoch}.pth')
    return avg_psnr

        
def train_safnet(config_path: str):
    config = Config(config_path)
    torch.manual_seed(config.base['seed'])
    if not config.base['debug']:
        init_wandb(config)
    train_loader = get_dataloader(config.data['train_dir'], config, is_train=True)
    test_loader = get_dataloader(config.data['test_dir'], config, is_train=False)
    model = get_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"Total Parameters": total_params, "Trainable Parameters": trainable_params})
    model = model.to(config.base['device'])
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    for epoch in range(config.train['num_epochs']):
        train_one_epoch(model, train_loader, optimizer,scheduler, config.base['device'], epoch, config)
        if epoch % config.train['val_freq'] == 0:
            validate_and_save(model, test_loader, config.base['device'], epoch, config)
if __name__ == '__main__':
    train_safnet('./configs/saf_config.yaml')