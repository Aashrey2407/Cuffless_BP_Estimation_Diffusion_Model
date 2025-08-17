import matplotlib.pyplot as plt
from data_pradyum import get_datasets
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from diffusion import BP_Diffusion
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
from model import DiffusionUNetCrossAttention, ConditionNet
import wandb

def set_deterministic(seed):
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
                      'which can slow down your training considerably!')

def freeze_model(model):
    print("Freezing model...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def load_model(PATH, nT, num_heads, epoch, device="cuda"):
    print("Loading models from checkpoints...")
    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    Conditioning_network1.load_state_dict(torch.load(PATH + f"ConditionNet1_epoch{epoch}.pth"))
    Conditioning_network2.load_state_dict(torch.load(PATH + f"ConditionNet2_epoch{epoch}.pth"))

    bp_diffusion = BP_Diffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2),
        n_T=nT
    )
    bp_diffusion.load_state_dict(torch.load(PATH + f"bp_diffusion_epoch{epoch}.pth"))

    Conditioning_network1 = freeze_model(Conditioning_network1)
    Conditioning_network2 = freeze_model(Conditioning_network2)
    bp_diffusion = freeze_model(bp_diffusion)

    return Conditioning_network1, Conditioning_network2, bp_diffusion

def evaluate_model(DATA_PATH, config, window_size, epoch):
    print("Starting ECG waveform evaluation only...")

    run = wandb.init(
        project="Cuffless_BP",
        entity="switchblade-bits-pilani",
        name=f"ecg_waveform_eval_epoch{epoch}",
        config=config,
        reinit=True
    )

    batch_size = config["batch_size"]
    nT = config["nT"]
    PATH = config["PATH"]
    device = config["device"]
    num_heads = config["attention_heads"]

    print("Loading datasets...")
    _, test_dataset = get_datasets(DATA_PATH, window_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    print("Loading and preparing models...")
    Conditioning_network1, Conditioning_network2, bp_diffusion = load_model(PATH, nT, num_heads, epoch, device=device)

    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)
    bp_diffusion = nn.DataParallel(bp_diffusion)

    print("Sampling 5 random ECG waveform comparisons...")
    random_indices = sorted(random.sample(range(len(test_dataset)), 5))
    saved_count = 0

    with torch.no_grad():
        for step, (ecg, ppg, ecg_roi, bp) in enumerate(tqdm(test_data, desc="Eval Batches")):
            global_index = step * batch_size
            for i in range(ecg.size(0)):
                absolute_index = global_index + i
                if absolute_index in random_indices:
                    ecg_i = ecg[i].unsqueeze(0).float().to(device)
                    ppg_i = ppg[i].unsqueeze(0).float().to(device)
                    ecg_roi_i = ecg_roi[i].unsqueeze(0).float().to(device)

                    cond1 = Conditioning_network1(ppg_i)
                    cond2 = Conditioning_network2(ppg_i)

                    gen_ecg = bp_diffusion(x=ecg_i, cond1=cond1, cond2=cond2, patch_labels=ecg_roi_i, mode="sample")

                    true_sig = ecg_i.cpu().numpy().flatten()
                    gen_sig = gen_ecg.cpu().numpy().flatten()

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(true_sig, label="True ECG")
                    ax.plot(gen_sig, label="Generated ECG", alpha=0.75)
                    ax.set_title(f"Sample {saved_count+1}: True vs Generated ECG")
                    ax.set_xlabel("Time step")
                    ax.set_ylabel("Amplitude")
                    ax.legend()
                    wandb.log({f"ecg_sample_{saved_count+1}": wandb.Image(fig)}, commit=False)
                    plt.close(fig)

                    saved_count += 1
                    if saved_count == 5:
                        run.finish()
                        print("Logged 5 samples. Done.")
                        return

if __name__ == "__main__":
    print("Setting deterministic mode...")
    set_deterministic(31)

    config = {
        "n_epoch": 1000,
        "batch_size": 2,
        "nT": 10,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 1,
        "alpha3": 1,
        "alpha4": 1,
        "PATH": "./"
    }

    DATA_PATH = "./preprocessed_mimic"
    print("Launching ECG waveform evaluation...")
    evaluate_model(DATA_PATH, config, 10, 174)

