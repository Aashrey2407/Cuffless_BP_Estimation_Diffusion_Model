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
from datetime import datetime
from bilstm import BP_Estimator

def freeze_model(model):
    print("Freezing model...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
def load_model(epoch, PATH, nT, num_heads, device="cuda"):
    device = torch.device(device)
    con1 = ConditionNet().to(device)
    con2 = ConditionNet().to(device)

    con1.load_state_dict(torch.load(PATH + f"ConditionNet1_epoch{epoch}.pth", map_location=device))
    con2.load_state_dict(torch.load(PATH + f"ConditionNet2_epoch{epoch}.pth", map_location=device))

    bp_diffusion = BP_Diffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2),
        n_T=nT
    )
    bp_diffusion.load_state_dict(torch.load(PATH + f"bp_diffusion_epoch{epoch}.pth", map_location=device))
    bp_diffusion = bp_diffusion.to(device)

    return con1, con2, bp_diffusion

def train_bilstm(config,resume_from_epoch = 0):
    print("We are inside train_bilstm")
    lr = 1e-3
    
    window_size = config["window_size"]
    PATH = config["PATH"]
    SAVE_PATH = config["SAVE_PATH"]
    nT = config["nT"]
    num_heads = config["num_heads"]
    batch_size = config["batch_size"]
    num_epochs = config["n_epochs"]
    device = config["device"]

    wandb.init(project="Cuffless_BP_Diffusion", config=config)  # ADDED
    print("wandb initiated")
    con1,con2,bp_diffusion = load_model(epoch = resume_from_epoch,PATH = PATH,nT=nT,num_heads = num_heads,device = device)

    con1 = freeze_model(con1)
    con2 = freeze_model(con2)
    bp_diffusion = freeze_model(bp_diffusion)

    bp_estimator = BP_Estimator(input_dim = 2,hidden_size = 128,num_layers = 2,output_dim = 2,dropout = 0.2).to(device)
    optimizer = torch.optim.Adam(bp_estimator.parameters(),lr = lr)

    train_data,_ = get_datasets(DATA_PATH,window_size)
    train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = True,num_workers = 32)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", position=0):
        bp_estimator.train()
        running_loss = 0.0

        for step, (ecg, ppg, ecg_roi, bp) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            ecg = ecg.float().to(device)
            ppg = ppg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)
            bp_true = bp.float().to(device)

            # Generate ECG from frozen diffusion model
            with torch.no_grad():
                cond1 = con1(ppg)
                cond2 = con2(ppg)
                gen_ecg = bp_diffusion(x=ecg, cond1=cond1, cond2=cond2, patch_labels=ecg_roi, mode="sample")

            # Train BP_Estimator
            optimizer.zero_grad()
            bp_pred, loss = bp_estimator(gen_ecg, ppg, bp_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        wandb.log({"epoch": epoch+1, "loss": avg_loss})  # ADDED

        if epoch%5==0:
            torch.save(bp_estimator.state_dict(), f"{SAVE_PATH}/bp_estimator_epoch{epoch+1}.pth")

wandb.finish()  # ADDED


            

if __name__ == "__main__":
    config = {
        "n_epochs": 1000,
        "batch_size": 2,
        "nT": 10,
        "device": "cuda",
        "num_heads": 8,
        "cond_mask": 0.0,
        "window_size": 4,
        "alpha1": 100,
        "alpha2": 1,
        "alpha3": 1,
        "alpha4": 1,
        "SAVE_PATH": "./checkpoints4sec_bilstm_aug13",
	"PATH": "../../pradyum/Cuffless_BP_Estimation_Diffusion_Model/checkpoints4sec_run_without_flip/"
    }

    DATA_PATH = "../../hemanth/Cuffless_BP_Estimation_Diffusion_Model/preprocessed_mimic"
    train_bilstm(config,57)




