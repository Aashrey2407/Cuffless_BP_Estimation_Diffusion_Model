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
from datetime import datetime
timenow=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
import os
# Configure PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.multiprocessing.set_sharing_strategy('file_system')
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

def train_rddm(config, resume_from_epoch=0):
    index = 0
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

    os.makedirs(PATH, exist_ok=True)
    start_epoch = resume_from_epoch if resume_from_epoch == 0 else resume_from_epoch + 1
    total_epochs = n_epoch

    wandb_resume_status = "allow" if resume_from_epoch > 0 else "never"
    wandb_run_name = f"HPC run {timenow}"
    if resume_from_epoch > 0:
        wandb_run_name += f" (resumed from epoch {resume_from_epoch})"

    wandb.init(
        project="Cuffless_BP_Diffusion",
        name=wandb_run_name,
        entity="somebbody-bits-pilani",
        resume=wandb_resume_status,
        config=config
    )

    global_step = resume_from_epoch  # ✅ Track wandb step globally

    dataset_train, dataset_test = get_datasets(
        DATA_PATH="./preprocessed_mimic",
        window_size=4
    )

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=32)

    bp_diffusion = BP_Diffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2),
        n_T=nT
    )
    bp_criterion = nn.MSELoss()
    # bp_estimator = BP_Estimator().to(device)

    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    bp_diffusion.to(device)

    if resume_from_epoch > 0:
        print(f"Loading checkpoints from epoch {resume_from_epoch}")
        try:
            bp_diffusion.load_state_dict(torch.load(f"{PATH}/bp_diffusion_epoch{resume_from_epoch}.pth"))
            Conditioning_network1.load_state_dict(torch.load(f"{PATH}/ConditionNet1_epoch{resume_from_epoch}.pth"))
            Conditioning_network2.load_state_dict(torch.load(f"{PATH}/ConditionNet2_epoch{resume_from_epoch}.pth"))
            # bp_estimator.load_state_dict(torch.load(f"{PATH}/bp_estimator_epoch{resume_from_epoch}.pth"))
            print(f"Successfully loaded all checkpoints from epoch {resume_from_epoch}")
            print(f"Training will continue from epoch {start_epoch}")
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")
            return

    optim = torch.optim.AdamW([*bp_diffusion.parameters(),
                               *Conditioning_network1.parameters(),
                               *Conditioning_network2.parameters()],
                            #    *bp_estimator.parameters()],
                              lr=1e-4)

    bp_diffusion = nn.DataParallel(bp_diffusion)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)
    # bp_estimator = nn.DataParallel(bp_estimator)

    scheduler = CosineAnnealingLRWarmup(optim, T_max=1000, T_warmup=20)

    for j in range(resume_from_epoch):
        scheduler.step()

    for i in range(start_epoch, total_epochs + start_epoch):
        current_epoch = i
        print(f"\n****************** Epoch - {current_epoch} *******************\n\n")

        bp_diffusion.train()
        Conditioning_network1.train()
        Conditioning_network2.train()
        # bp_estimator.train()

        total_loss = 0
        total_ddpm = 0
        total_region = 0
        total_align = 0
        # total_bp = 0
        num_batches = 0

        pbar = tqdm(dataloader)

        for y_ecg, x_ppg, ecg_roi, bp in pbar:
            if index == 0:
                print(f"Ecg: {y_ecg.shape} Ppg: {x_ppg.shape} EcgRoi: {ecg_roi.shape} Bp: {bp.shape}")
            index += 1

            optim.zero_grad()
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)
            bp = bp.float().to(device)

            ppg_conditions1 = Conditioning_network1(x_ppg)
            ppg_conditions2 = Conditioning_network2(x_ppg)

            ddpm_loss, region_loss, alignment_loss, generated_ecg = bp_diffusion(
                x=y_ecg, cond1=ppg_conditions1, cond2=ppg_conditions2, patch_labels=ecg_roi
            )
            # bp_pred = bp_estimator(generated_ecg, x_ppg)
            # bp_loss = bp_criterion(bp_pred, bp)

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            alignment_loss = alpha3 * alignment_loss
            # bp_loss = alpha4 * bp_loss
            loss = ddpm_loss + region_loss + alignment_loss #+ bp_loss

            loss.mean().backward()
            optim.step()

            total_loss += loss.mean().item()
            total_ddpm += ddpm_loss.mean().item()
            total_region += region_loss.mean().item()
            total_align += alignment_loss.mean().item()
            # total_bp += bp_loss.mean().item()
            num_batches += 1

            pbar.set_description(f"loss: {loss.mean().item():.4f}")

        scheduler.step()

        # ✅ Log to wandb ONCE per epoch (averaged over all batches)
        wandb.log({
            "epoch": current_epoch,
            "loss": total_loss / num_batches,
            "DDPM_loss": total_ddpm / num_batches,
            "Region_loss": total_region / num_batches,
            "Alignment_loss": total_align / num_batches,
            # "BP_loss": total_bp / num_batches,
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=global_step)

        global_step += 1
        print(f"Current epoch : {current_epoch}")
        print(f"average loss : {total_loss/num_batches}")
        print(f"ddpm loss : {total_ddpm/num_batches}")
        print(f"region loss : {total_region/num_batches}")
        print(f"alignment loss : {total_align/num_batches}")
        if current_epoch % 5 == 0:
            wandb.log({
                "epoch_completed": current_epoch,
                "epochs_remaining": total_epochs + start_epoch - current_epoch - 1
            })
        torch.save(bp_diffusion.module.state_dict(), f"{PATH}/bp_diffusion_epoch{current_epoch}.pth")
        torch.save(Conditioning_network1.module.state_dict(), f"{PATH}/ConditionNet1_epoch{current_epoch}.pth")
        torch.save(Conditioning_network2.module.state_dict(), f"{PATH}/ConditionNet2_epoch{current_epoch}.pth")
        # torch.save(bp_estimator.state_dict(), f"{PATH}/bp_estimator_epoch{current_epoch}.pth")
        wandb.save(f"{PATH}/bp_diffusion_epoch{current_epoch}.pth")
        wandb.save(f"{PATH}/ConditionNet1_epoch{current_epoch}.pth")
        wandb.save(f"{PATH}/ConditionNet2_epoch{current_epoch}.pth")
        # wandb.save(f"{PATH}/bp_estimator_epoch{current_epoch}.pth")

        wandb.log({"checkpoint_epoch": current_epoch})
        print(f"Saved checkpoint for epoch {current_epoch}")

                
if __name__ == "__main__":
    config = {
        "n_epoch": 1000,
        "batch_size": 16,
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "alpha3": 1,  # Weight for alignment loss
        "alpha4": 1,  # Weight for BP loss
        "PATH": "./checkpoints4sec_ppg_to_ecg_Aug12"
    }

    train_rddm(config, resume_from_epoch=0)
