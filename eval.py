import matplotlib.pyplot as plt
from data_pradyum import get_datasets
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from diffusion import load_pretrained_DPM, BP_Diffusion
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
from model import DiffusionUNetCrossAttention, ConditionNet, BP_Estimator
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
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

def freeze_model(model):
    print("Freezing model...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def _c(i, j, P, Q, ca):
    if ca[i, j] > -1:
        return ca[i, j]
    d = abs(P[i] - Q[j])
    if i == 0 and j == 0:
        ca[i, j] = d
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(i-1, 0, P, Q, ca), d)
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(0, j-1, P, Q, ca), d)
    else:
        ca[i, j] = max(
            min(
                _c(i-1, j,   P, Q, ca),
                _c(i-1, j-1, P, Q, ca),
                _c(i,   j-1, P, Q, ca)
            ),
            d
        )
    return ca[i, j]

def discrete_frechet_distance(P, Q):
    # P, Q: 1D numpy arrays
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    return _c(n-1, m-1, P, Q, ca)

def load_model(PATH, nT, num_heads, epoch, device="cuda"):
    print("Loading models from checkpoints...")
    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    Conditioning_network1.load_state_dict(torch.load(PATH + f"ConditionNet1_epoch{epoch}.pth"))
    Conditioning_network2.load_state_dict(torch.load(PATH + f"ConditionNet2_epoch{epoch}.pth"))

    # bp_estimator = BP_Estimator().to(device)
    # bp_estimator.load_state_dict(torch.load(PATH + f"bp_estimator_epoch{epoch}.pth"))

    bp_diffusion = BP_Diffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2),
        n_T=nT
    )
    bp_diffusion.load_state_dict(torch.load(PATH + f"bp_diffusion_epoch{epoch}.pth"))

    print("Freezing models for evaluation...")
    Conditioning_network1 = freeze_model(Conditioning_network1)
    Conditioning_network2 = freeze_model(Conditioning_network2)
    bp_diffusion = freeze_model(bp_diffusion)
    # bp_estimator = freeze_model(bp_estimator)

    return Conditioning_network1, Conditioning_network2, bp_diffusion#, bp_estimator

def evaluate_model(DATA_PATH, config, window_size, epoch):
    print("Starting evaluation...")

    run = wandb.init(
        project="Cuffless_BP",
        entity="f20220160-bits-pilani",
        name=f"evaluation_run_for_epoch{epoch}",
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
    random_indices = random.sample(range(len(test_dataset)), 5)

    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    print("Loading and preparing models...")
    # Conditioning_network1, Conditioning_network2, bp_diffusion, bp_estimator = load_model(PATH, nT, num_heads, epoch, device=device)
    Conditioning_network1, Conditioning_network2, bp_diffusion = load_model(PATH, nT, num_heads, epoch, device=device)

    print("Wrapping models with DataParallel...")
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)
    bp_diffusion = nn.DataParallel(bp_diffusion)
    # bp_estimator = nn.DataParallel(bp_estimator)

    predictions = []
    true_values = []

    print("Beginning evaluation loop...")
    with torch.no_grad():
        for step, (ecg, ppg, ecg_roi, bp) in enumerate(tqdm(test_data, desc="Eval Batches")):
            ecg = ecg.float().to(device)
            ppg = ppg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)
            bp_true = bp.float().to(device)

            cond1 = Conditioning_network1(ppg)
            cond2 = Conditioning_network2(ppg)

            gen_ecg = bp_diffusion(x=ecg, cond1=cond1, cond2=cond2, patch_labels=ecg_roi, mode="sample")

            # bp_pred = bp_estimator(gen_ecg, ppg)

            # predictions.append(bp_pred.cpu().numpy())
            # true_values.append(bp_true.cpu().numpy())

            global_index = step * batch_size
            for i in range(ecg.size(0)):
                current_index = global_index + i
                if current_index in random_indices:
                    print(f"Logging ECG waveform comparison for index {current_index}...")
                    true_sig = ecg[i].cpu().numpy().flatten()
                    gen_sig = gen_ecg[i].cpu().numpy().flatten()
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(true_sig, label="True ECG")
                    ax.plot(gen_sig, label="Generated ECG", alpha=0.75)
                    ax.set_title(f"True vs Generated Signal (Index {current_index})")
                    ax.set_xlabel("Time step")
                    ax.set_ylabel("Amplitude")
                    ax.legend()
                    #fd = discrete_frechet_distance(true_sig,gen_sig)
                    #print(f"Frechet distance for index {current_index} is : {fd}")
                    wandb.log({f"ecg_waveform_comparison_{current_index}": wandb.Image(fig)}, commit=False)
                    plt.close(fig)

    # print("Calculating metrics...")
    # preds = np.vstack(predictions)
    # trues = np.vstack(true_values)

    # sbp_pred, dbp_pred = preds[:, 0], preds[:, 1]
    # sbp_true, dbp_true = trues[:, 0], trues[:, 1]

    # mae_sbp = np.mean(np.abs(sbp_pred - sbp_true))
    # rmse_sbp = np.sqrt(np.mean((sbp_pred - sbp_true) ** 2))

    # mae_dbp = np.mean(np.abs(dbp_pred - dbp_true))
    # rmse_dbp = np.sqrt(np.mean((dbp_pred - dbp_true) ** 2))

    # print("Logging metrics to Weights & Biases...")
    # wandb.log({
    #     "MAE_SBP": mae_sbp,
    #     "RMSE_SBP": rmse_sbp,
    #     "MAE_DBP": mae_dbp,
    #     "RMSE_DBP": rmse_dbp
    # })

    run.finish()

    # print("\n--- Final Results ---")
    # print(f"SBP → MAE: {mae_sbp:.3f}, RMSE: {rmse_sbp:.3f}")
    # print(f"DBP → MAE: {mae_dbp:.3f}, RMSE: {rmse_dbp:.3f}")
    # print("---------------------")


if __name__ == "__main__":
    print("Setting deterministic mode...")
   # set_deterministic(31)

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
        "PATH": "./checkpoints4sec_ppg_to_ecg_without_flip_16Aug/"
    }

    DATA_PATH = "../../hemanth/Cuffless_BP_Estimation_Diffusion_Model/preprocessed_mimic"
    print("Launching evaluation...")
    evaluate_model(DATA_PATH, config,4,57)
