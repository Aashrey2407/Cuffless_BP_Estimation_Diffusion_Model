import torch
import wandb
import random
from tqdm import tqdm
import warnings
import numpy as np
import os
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from data_pradyum import get_datasets
import torch.nn as nn
from metrics import *
from lr_scheduler import CosineAnnealingLRWarmup
from torch.utils.data import DataLoader

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_cached() / 1e9:.2f} GB")

def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_rddm(config, resume_from_epoch=0):
    device = config["device"]
    scaler = torch.cuda.amp.GradScaler()
    
    dataset_train, dataset_test = get_datasets(
        DATA_PATH="./preprocessed_mimic",
        window_size=10
    )
    
    dataloader = DataLoader(
        dataset_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=config["attention_heads"]),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=config["attention_heads"]),
        betas=(1e-4, 0.2),
        n_T=config["nT"]
    )
    
    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    rddm.to(device)
    
    if resume_from_epoch > 0:
        print(f"Loading checkpoints from epoch {resume_from_epoch}")
        rddm.load_state_dict(torch.load(f"{config['PATH']}/RDDM_epoch{resume_from_epoch}.pth"))
        Conditioning_network1.load_state_dict(torch.load(f"{config['PATH']}/ConditionNet1_epoch{resume_from_epoch}.pth"))
        Conditioning_network2.load_state_dict(torch.load(f"{config['PATH']}/ConditionNet2_epoch{resume_from_epoch}.pth"))
    
    optim = torch.optim.AdamW(
        [*rddm.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters()],
        lr=1e-4
    )
    
    rddm = nn.DataParallel(rddm)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)
    
    scheduler = CosineAnnealingLRWarmup(optim, T_max=1000, T_warmup=20)
    
    for epoch in range(resume_from_epoch, config["n_epoch"]):
        print(f"\n****************** Epoch - {epoch} *******************\n")
        print_gpu_memory()
        
        rddm.train()
        Conditioning_network1.train()
        Conditioning_network2.train()
        
        pbar = tqdm(dataloader)
        for step, (y_ecg, x_ppg, ecg_roi) in enumerate(pbar):
            with torch.cuda.amp.autocast():
                optim.zero_grad()
                
                x_ppg = x_ppg.float().to(device)
                y_ecg = y_ecg.float().to(device)
                ecg_roi = ecg_roi.float().to(device)
                
                ppg_conditions1 = Conditioning_network1(x_ppg)
                ppg_conditions2 = Conditioning_network2(x_ppg)
                
                ddpm_loss, region_loss = rddm(
                    x=y_ecg,
                    cond1=ppg_conditions1,
                    cond2=ppg_conditions2,
                    patch_labels=ecg_roi
                )
                
                ddpm_loss = config["alpha1"] * ddpm_loss
                region_loss = config["alpha2"] * region_loss
                loss = ddpm_loss + region_loss
            
            scaler.scale(loss.mean()).backward()
            scaler.step(optim)
            scaler.update()
            
            if step % 2 == 0:  # Gradient accumulation steps
                torch.cuda.empty_cache()
            
            pbar.set_description(f"loss: {loss.mean().item():.4f}")
            
            wandb.log({
                "epoch": epoch,
                "loss": loss.mean().item(),
                "DDPM_loss": ddpm_loss.mean().item(),
                "Region_loss": region_loss.mean().item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        scheduler.step()
        
        if epoch % 80 == 0:
            torch.save(rddm.module.state_dict(), f"{config['PATH']}/RDDM_epoch{epoch}.pth")
            torch.save(Conditioning_network1.module.state_dict(), f"{config['PATH']}/ConditionNet1_epoch{epoch}.pth")
            torch.save(Conditioning_network2.module.state_dict(), f"{config['PATH']}/ConditionNet2_epoch{epoch}.pth")
            wandb.save(f"{config['PATH']}/RDDM_epoch{epoch}.pth")
            wandb.log({"checkpoint_epoch": epoch})

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    config = {
        "n_epoch": 1000,
        "batch_size": 16,  # Reduced batch size
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "PATH": "./"
    }
    
    set_deterministic(31)
    
    wandb.init(
        project="Cuffless_BP",
        name="RDDM_training_optimized",
        entity="switchblade-bits-pilani",
        config=config
    )
    
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    
    train_rddm(config, resume_from_epoch=0)
