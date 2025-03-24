import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import BPDiffusion  # Changed from RDDM to BPDiffusion
from diffusion import BP_Estimator
from data_modified import get_datasets_physionet
import torch.nn as nn
from metrics import *
# from torch.optim.lr_scheduler import CosineAnnealingLRWarmup
from torch.utils.data import Dataset, DataLoader

class CosineAnnealingLRWarmup:
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda step: min(1.0, step / T_warmup)
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max - T_warmup,
            eta_min=eta_min
        )
        self.current_step = 0

    def step(self):
        if self.current_step < self.T_warmup:
            self.warmup_scheduler.step()
        else:
            # Adjust step count for the cosine scheduler to start from zero after warmup
            self.cosine_scheduler.step(self.current_step - self.T_warmup)
        self.current_step += 1

    def get_last_lr(self):
        if self.current_step < self.T_warmup:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.cosine_scheduler.get_last_lr()



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
    
    data_dir = "./patient"
    record_list = ["3141595_0001"]  # List of record names (without file extensions)
    dataset_train = get_datasets_physionet(data_dir, record_list)

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    print(dataloader)

    # Instantiate the new BPDiffusion model
    bp_diffusion = BPDiffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT,
        lam1=config["lam1"],    # e.g., 100
        lam2=config["lam2"],    # e.g., 1
        sampling_rate=128
    ).to(device)

    bp_estimator = BP_Estimator(input_dim = 2,hidden_size = 128,num_layers = 2,output_dim = 2,dropout = 0.2).to(device)


    # Use a single ConditionNet instance for unified conditioning
    Conditioning_network = ConditionNet().to(device)
    
    bp_diffusion.to(device)

    optim = torch.optim.AdamW(
        [*bp_diffusion.parameters(), *Conditioning_network.parameters(), *bp_estimator.parameters()], lr=1e-4
    )

    bp_diffusion = nn.DataParallel(bp_diffusion)
    Conditioning_network = nn.DataParallel(Conditioning_network)
    bp_estimator = nn.DataParallel(bp_estimator)
    scheduler = CosineAnnealingLRWarmup(optim, T_max=1000,T_warmup=20,eta_min=1e-6)
    
    for i in range(n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n")
        bp_diffusion.train()
        Conditioning_network.train()
        bp_estimator.train()
        pbar = tqdm(dataloader)

        for y_ecg, x_ppg, ecg_roi,bp_target in pbar:
            optim.zero_grad()
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)
            bp_target = bp_target.float().to(device)
            # Get unified conditioning from the PPG signal
            ppg_conditions = Conditioning_network(x_ppg, drop_prob=cond_mask)

            # Forward pass with BPDiffusion; it returns (total_loss, L_q, L_a)
            total_loss, L_q, L_a = bp_diffusion(x=y_ecg, cond=ppg_conditions, patch_labels=ecg_roi, mode="train")
            gen_ecg = bp_diffusion(x=y_ecg,cond=ppg_conditions,patch_labels = ecg_roi,mode = "sample")

            bp_pred,L_bp = bp_estimator(gen_ecg,x_ppg,bp_target)
            overall_loss = total_loss + L_bp

            overall_loss.mean().backward()
            optim.step()

            pbar.set_description(f"Total Loss: {overall_loss.mean().item():.4f}, L_q: {L_q.mean().item():.4f}, L_a: {L_a:.4f},L_bp: {L_bp.mean().item():.4f}")

            wandb.log({
                "Overall_loss": overall_loss.mean().item(),
                "L_q": L_q.mean().item(),
                "L_a": L_a,
                "L_bp": L_bp.mean().item(),
            })

        scheduler.step()

        if i % 80 == 0:
            torch.save(bp_diffusion.module.state_dict(), f"{PATH}/BPDiffusion_epoch{i}.pth")
            torch.save(Conditioning_network.module.state_dict(), f"{PATH}/ConditionNet_epoch{i}.pth")
            torch.save(bp_estimator.module.state_dict(), f"{PATH}/BP_Estimator_epoch{i}.pth")

if __name__ == "__main__":
    config = {
        "n_epoch": 1000,
        "batch_size": 128*4,
        "nT": 10,
        "device": "cpu",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "lam1": 100,   # Weight for QRS region noise loss
        "lam2": 1,     # Weight for reconstruction loss (if region_model is used)
        "PATH": "./"
    }
    train_bp_diffusion(config)
