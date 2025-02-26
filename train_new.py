import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import BPDiffusion  # Changed from RDDM to BPDiffusion
from data import get_datasets
import torch.nn as nn
from metrics import *
from lr_scheduler import CosineAnnealingLRWarmup
from torch.utils.data import Dataset, DataLoader

def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('Seeding training. This will turn on deterministic settings which may slow down training.')

set_deterministic(31)

def train_bp_diffusion(config):
    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    PATH = config["PATH"]

    wandb.init(
        project="INSERT PROJECT NAME HERE",
        entity="INSERT ENTITY HERE",
        id=f"INSERT ID HERE",
        config=config
    )

    dataset_train, _ = get_datasets()

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=128)

    # Instantiate the new BPDiffusion model
    bp_diffusion = BPDiffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT,
        lam1=config["lam1"],    # e.g., 100
        lam2=config["lam2"],    # e.g., 1
        sampling_rate=128
    ).to(device)

    # Use a single ConditionNet instance for unified conditioning
    Conditioning_network = ConditionNet().to(device)
    
    bp_diffusion.to(device)

    optim = torch.optim.AdamW(
        [*bp_diffusion.parameters(), *Conditioning_network.parameters()], lr=1e-4
    )

    bp_diffusion = nn.DataParallel(bp_diffusion)
    Conditioning_network = nn.DataParallel(Conditioning_network)

    scheduler = CosineAnnealingLRWarmup(optim, T_max=1000, T_warmup=20)
    
    for i in range(n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n")
        bp_diffusion.train()
        Conditioning_network.train()
        pbar = tqdm(dataloader)

        for y_ecg, x_ppg, ecg_roi in pbar:
            optim.zero_grad()
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            # Get unified conditioning from the PPG signal
            ppg_conditions = Conditioning_network(x_ppg, drop_prob=cond_mask)

            # Forward pass with BPDiffusion; it returns (total_loss, L_q, L_a)
            total_loss, L_q, L_a = bp_diffusion(x=y_ecg, cond=ppg_conditions, patch_labels=ecg_roi, mode="train")
            
            total_loss.mean().backward()
            optim.step()

            pbar.set_description(f"Total Loss: {total_loss.mean().item():.4f}, L_q: {L_q.mean().item():.4f}, L_a: {L_a:.4f}")

            wandb.log({
                "Total_loss": total_loss.mean().item(),
                "L_q": L_q.mean().item(),
                "L_a": L_a,
            })

        scheduler.step()

        if i % 80 == 0:
            torch.save(bp_diffusion.module.state_dict(), f"{PATH}/BPDiffusion_epoch{i}.pth")
            torch.save(Conditioning_network.module.state_dict(), f"{PATH}/ConditionNet_epoch{i}.pth")

if __name__ == "__main__":
    config = {
        "n_epoch": 1000,
        "batch_size": 128*4,
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "lam1": 100,   # Weight for QRS region noise loss
        "lam2": 1,     # Weight for reconstruction loss (if region_model is used)
        "PATH": "INSERT PATH HERE"
    }
    train_bp_diffusion(config)
