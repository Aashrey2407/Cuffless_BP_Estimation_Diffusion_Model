import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from diffusion import BP_Diffusion
from data_pradyum import get_datasets
import torch.nn as nn
from metrics import *
from lr_scheduler import CosineAnnealingLRWarmup
from torch.utils.data import Dataset, DataLoader
from model import BP_Estimator

import os
# Configure PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

set_deterministic(31)

def train_rddm(config, resume_from_epoch=800):

    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    alpha1 = config["alpha1"]
    alpha2 = config["alpha2"]
    alpha3 = config["alpha3"]
    alpha4 = config["alpha4"]
    PATH = config["PATH"]

    wandb.init(
        project="Cuffless_BP",
        name=f"RDDM_training_resumed_from_epoch_{resume_from_epoch}",
        entity="switchblade-bits-pilani",
        id=None,
        resume="allow",  # This allows resuming if the run ID already exists
        config=config
    )

    dataset_train, dataset_test = get_datasets(
    DATA_PATH="./preprocessed_mimic",  # Path where preprocessed_mimic directory is located
    window_size=10
)

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=128)

    '''rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT
    )'''
    bp_diffusion = BP_Diffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT
    )
    bp_criterion = nn.MSELoss()
    bp_estimator = BP_Estimator().to(device)

    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    bp_diffusion.to(device)

    if resume_from_epoch > 0:
        print(f"Loading checkpoints from epoch {resume_from_epoch}")
        bp_diffusion.load_state_dict(torch.load(f"{PATH}/BP_Diffusion_epoch{resume_from_epoch}.pth"))
        Conditioning_network1.load_state_dict(torch.load(f"{PATH}/ConditionNet1_epoch{resume_from_epoch}.pth"))
        Conditioning_network2.load_state_dict(torch.load(f"{PATH}/ConditionNet2_epoch{resume_from_epoch}.pth"))



    optim = torch.optim.AdamW([*bp_diffusion.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters(), *bp_estimator.parameters()], lr=1e-4)

    bp_diffusion = nn.DataParallel(bp_diffusion)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    scheduler = CosineAnnealingLRWarmup(optim, T_max=1000, T_warmup=20)
    for j in range(resume_from_epoch):
        scheduler.step()
    for i in range(0,n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n\n")

        bp_diffusion.train()
        Conditioning_network1.train()
        Conditioning_network2.train()
        pbar = tqdm(dataloader)

        for y_ecg, x_ppg, ecg_roi,bp in pbar:
            
            ## Train Diffusion
            optim.zero_grad()
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            ppg_conditions1 = Conditioning_network1(x_ppg)
            ppg_conditions2 = Conditioning_network2(x_ppg)

            '''ddpm_loss, region_loss = rddm(x=y_ecg, cond1=ppg_conditions1, cond2=ppg_conditions2, patch_labels=ecg_roi)

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            
            loss = ddpm_loss + region_loss'''
            ddpm_loss, region_loss, alignment_loss,generated_ecg = bp_diffusion(
                x=y_ecg, cond1=ppg_conditions1, cond2=ppg_conditions2, patch_labels=ecg_roi
            )
            bp_pred = bp_estimator(generated_ecg, x_ppg)
            bp_loss = bp_criterion(bp_pred, bp.to(device))

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            alignment_loss = alpha3 * alignment_loss
            bp_loss = alpha4 * bp_loss
            loss = ddpm_loss + region_loss + alignment_loss + bp_loss

            loss.mean().backward()
            
            optim.step()

            pbar.set_description(f"loss: {loss.mean().item():.4f}")

            # In your training loop, after calculating losses
            '''wandb.log({
                "epoch": i,
                "loss": loss.mean().item(),
                "DDPM_loss": ddpm_loss.mean().item(),
                "Region_loss": region_loss.mean().item(),
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=i)'''
            wandb.log({
                "epoch": i,
                "loss": loss.mean().item(),
                "DDPM_loss": ddpm_loss.mean().item(),
                "Region_loss": region_loss.mean().item(),
                "Alignment_loss": alignment_loss.mean().item(),
                "BP_loss": bp_loss.mean().item(),
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=i)

        scheduler.step()
        # Add this after your training loop for each epoch
        if i % 5 == 0:  # Every 10 epochs
                # Log example predictions if applicable
            wandb.log({
                "epoch_completed": i,
                "epochs_remaining": n_epoch - i
            })
        if i % 80 == 0:
            torch.save(bp_diffusion.module.state_dict(), f"{PATH}/bp_diffusion_epoch{i}.pth")
            torch.save(Conditioning_network1.module.state_dict(), f"{PATH}/ConditionNet1_epoch{i}.pth")
            torch.save(Conditioning_network2.module.state_dict(), f"{PATH}/ConditionNet2_epoch{i}.pth")
            wandb.save(f"{PATH}/bp_diffusion_epoch{i}.pth")
            wandb.log({"checkpoint_epoch": i})

                
if __name__ == "__main__":

    '''config = {
        "n_epoch": 1000,
        "batch_size":2,
        "nT":10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "PATH": "./"
    }'''
    config = {
        "n_epoch": 1000,
        "batch_size": 2,
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "alpha3": 1,  # New weight for alignment loss
        "alpha4": 1,  # New weight for BP loss
        "PATH": "./"
    }

    train_rddm(config, resume_from_epoch=0)
