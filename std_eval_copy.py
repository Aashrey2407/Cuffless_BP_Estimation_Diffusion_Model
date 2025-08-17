import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM  # (We will no longer use this function for loading checkpoints.)
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_pradyum import get_datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from diffusion import BPDiffusion  # Changed from RDDM to BPDiffusion

# Configure PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cuda.matmul.allow_tf32 = True

# Create a directory to save plots if it doesn't exist
os.makedirs("ecg_plots", exist_ok=True)

# NEW: Import the new BP_Estimator and unified ConditionNet from model
from model import DiffusionUNetCrossAttention, ConditionNet, BP_Estimator

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
set_deterministic(31)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

# CHANGED: Updated plotting function to include predicted and true BP info in titles.
def plot_ecg_comparison(real_ecg, fake_ecg, ppg_signal=None, roi=None, predicted_bp=None, true_bp=None, sample_idx=0, seconds=5):
    """
    Plot real vs generated ECG along with PPG and ROI (if available)
    and annotate with BP estimation results.
    """
    sample_rate = 128  # Hz
    n_samples = int(seconds * sample_rate)
    real_segment = real_ecg[sample_idx, :n_samples]
    fake_segment = fake_ecg[sample_idx, :n_samples]
    time = np.arange(n_samples) / sample_rate
    
    plt.figure(figsize=(15, 10))
    
    # Number of subplots: always 2 (ECG traces) + one each for PPG and ROI if provided.
    n_plots = 2  
    if ppg_signal is not None:
        n_plots += 1
    if roi is not None:
        n_plots += 1
    
    # Plot real ECG with true BP info if provided.
    plt.subplot(n_plots, 1, 1)
    title_str = "Real ECG"
    if (predicted_bp is not None) and (true_bp is not None):
        title_str += f" | True BP: SBP {true_bp[sample_idx,0]:.1f}, DBP {true_bp[sample_idx,1]:.1f}"
    plt.plot(time, real_segment, 'b-', linewidth=1.5, label='Real ECG')
    plt.title(title_str)
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot generated ECG with predicted BP info if provided.
    plt.subplot(n_plots, 1, 2)
    title_str = "Generated ECG"
    if (predicted_bp is not None) and (true_bp is not None):
        title_str += f" | Predicted BP: SBP {predicted_bp[sample_idx,0]:.1f}, DBP {predicted_bp[sample_idx,1]:.1f}"
    plt.plot(time, fake_segment, 'r-', linewidth=1.5, label='Generated ECG')
    plt.title(title_str)
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    current_plot = 3
    if ppg_signal is not None:
        ppg_segment = ppg_signal[sample_idx, :n_samples]
        plt.subplot(n_plots, 1, current_plot)
        plt.plot(time, ppg_segment, 'g-', linewidth=1.5, label='PPG')
        plt.title('PPG Signal (Input)')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        current_plot += 1
    
    if roi is not None:
        roi_segment = roi[sample_idx, :n_samples]
        plt.subplot(n_plots, 1, current_plot)
        plt.plot(time, roi_segment, 'm-', linewidth=1.5, label='ROI')
        plt.title('Region of Interest')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    save_path = f"ecg_plots/ecg_comparison_sample_{sample_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ECG comparison plot saved to {save_path}")

# CHANGED: Updated evaluation function to load the new BP_Diffusion, unified ConditionNet, and BP_Estimator.
def eval_diffusion(window_size=10, nT=10, batch_size=8, device="cuda", PATH="./"):
    # Load test dataset from the preprocessed_mimic folder
    _, dataset_test = get_datasets("./preprocessed_mimic", window_size)
    # NEW: Now dataset returns 4 items (ECG, PPG, ROI, BP)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Instead of calling load_pretrained_DPM, we load the three modules manually
    # NEW: Load new BP_Diffusion model checkpoint
    bp_diffusion = BPDiffusion(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=config["attention_heads"]),
        betas=(1e-4, 0.2),
        n_T=nT,
        lam1=config["lam1"],    # e.g., 100
        lam2=config["lam2"],    # e.g., 1
        sampling_rate=128
    ).to(device)
    # Load the state dictionary (since checkpoints were saved via state_dict)
    bp_diffusion.load_state_dict(torch.load(PATH + "BPDiffusion_epoch_latest.pth", map_location=device))

    # Instantiate the unified ConditionNet (used for conditioning in training)
    Conditioning_network = ConditionNet().to(device)
    Conditioning_network.load_state_dict(torch.load(PATH + "ConditionNet_epoch_latest.pth", map_location=device))

    # Instantiate the BP_Estimator (BiLSTM model for BP estimation)
    bp_estimator = BP_Estimator(
        input_dim=2,         # Matching training: expecting concatenated (generated ECG, PPG)
        hidden_size=128,
        num_layers=2,
        output_dim=2,
        dropout=0.2
    ).to(device)
    bp_estimator.load_state_dict(torch.load(PATH + "BP_Estimator_epoch_latest.pth", map_location=device))

    
    # Wrap in DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        bp_diffusion = nn.DataParallel(bp_diffusion)
        cond_net = nn.DataParallel(cond_net)
        bp_estimator = nn.DataParallel(bp_estimator)
        
    bp_diffusion.eval()
    cond_net.eval()
    bp_estimator.eval()
    
    fd_list = []
    fake_ecgs = []
    real_ecgs = []
    real_ppgs = []
    true_rois = []
    all_bp_pred = []
    all_bp_true = []
    sample_generated_ecgs = []  # For visualization
    
    total_samples = 128 * window_size  # e.g., 1280 samples
    # NEW: Loop now unpacks BP from the dataset
    for y_ecg, x_ppg, ecg_roi, bp in tqdm(testloader):
        torch.cuda.empty_cache()
        x_ppg = x_ppg.float().to(device)
        y_ecg = y_ecg.float().to(device)
        ecg_roi = ecg_roi.float().to(device)
        bp = bp.float()  # bp shape: (batch, 2)
        
        generated_windows = []
        # Process PPG in chunks as before
        for ppg_window in torch.split(x_ppg, 128*4, dim=-1):
            if ppg_window.shape[-1] != 128*4:
                ppg_window = F.pad(ppg_window, (0, 128*4 - ppg_window.shape[-1]), "constant", 0)
            # Get conditioning from the unified ConditionNet
            ppg_conditions = cond_net(ppg_window)
            # Generate ECG using bp_diffusion in sample mode.
            # NEW: Ensure window_size passed is total_samples (e.g., 1280).
            xh = bp_diffusion(
                x=y_ecg, 
                cond=ppg_conditions, 
                patch_labels=ecg_roi, 
                mode="sample", 
                window_size=total_samples
            )
            generated_windows.append(xh.cpu().numpy())
            
        xh = np.concatenate(generated_windows, axis=-1)[:, :, :total_samples]
        
        fd = calculate_FD(y_ecg, torch.from_numpy(xh).to(device))
        fd_list.append(fd)
        
        fake_ecgs_batch = xh.reshape(-1, total_samples)
        real_ecgs_batch = y_ecg.reshape(-1, total_samples).cpu().numpy()
        real_ppgs_batch = x_ppg.reshape(-1, total_samples).cpu().numpy()
        true_rois_batch = ecg_roi.reshape(-1, total_samples).cpu().numpy()
        
        fake_ecgs.append(fake_ecgs_batch)
        real_ecgs.append(real_ecgs_batch)
        real_ppgs.append(real_ppgs_batch)
        true_rois.append(true_rois_batch)
        
        # NEW: BP Estimation. Pass generated ECG and x_ppg to bp_estimator.
        generated_ecg = torch.from_numpy(xh).to(device)  # shape: (batch, 1, total_samples)
        bp_pred = bp_estimator(generated_ecg, x_ppg)  # Expected shape: (batch, 2)
        all_bp_pred.append(bp_pred.cpu().numpy())
        all_bp_true.append(bp.cpu().numpy())
        
        # For visualization, store a sample from the first few batches.
        if len(sample_generated_ecgs) < 3:
            sample_generated_ecgs.append(generated_ecg[0].cpu().numpy())
    
    fake_ecgs = np.concatenate(fake_ecgs, axis=0)
    real_ecgs = np.concatenate(real_ecgs, axis=0)
    real_ppgs = np.concatenate(real_ppgs, axis=0)
    true_rois = np.concatenate(true_rois, axis=0)
    all_bp_pred = np.concatenate(all_bp_pred, axis=0)
    all_bp_true = np.concatenate(all_bp_true, axis=0)
    
    # NEW: Compute BP metrics separately for SBP and DBP.
    mae_sbp = np.mean(np.abs(all_bp_true[:, 0] - all_bp_pred[:, 0]))
    mae_dbp = np.mean(np.abs(all_bp_true[:, 1] - all_bp_pred[:, 1]))
    rmse_sbp = np.sqrt(np.mean((all_bp_true[:, 0] - all_bp_pred[:, 0]) ** 2))
    rmse_dbp = np.sqrt(np.mean((all_bp_true[:, 1] - all_bp_pred[:, 1]) ** 2))
    
    mean_fd = np.mean(fd_list)
    
    tracked_metrics = {
        "MAE_SBP": mae_sbp,
        "MAE_DBP": mae_dbp,
        "RMSE_SBP": rmse_sbp,
        "RMSE_DBP": rmse_dbp,
        "FD": mean_fd,
    }
    
    # NEW: Plot an ECG comparison including BP information.
    sample_idx = 0
    plot_ecg_comparison(
        real_ecg=real_ecgs, 
        fake_ecg=fake_ecgs, 
        ppg_signal=real_ppgs,
        roi=true_rois,
        predicted_bp=all_bp_pred,
        true_bp=all_bp_true,
        sample_idx=sample_idx,
        seconds=5
    )
    
    # Existing detailed error analysis plot remains unchanged.
    plt.figure(figsize=(20, 10))
    mse_over_time = (real_ecgs[sample_idx] - fake_ecgs[sample_idx])**2
    plt.subplot(3, 1, 1)
    plt.plot(real_ecgs[sample_idx][:640], 'b-', label='Real ECG')
    plt.title('Real vs Generated ECG (First 5 seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(fake_ecgs[sample_idx][:640], 'r-', label='Generated ECG')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(mse_over_time[:640], 'g-', label='Squared Error')
    plt.xlabel('Sample Index (128 Hz)')
    plt.ylabel('Squared Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ecg_plots/ecg_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Detailed error analysis plot saved to ecg_plots/ecg_error_analysis.png")
    return tracked_metrics

if __name__ == "__main__":
    # Updated config with new keys and checkpoint path
    config = {
        "batch_size": 8,               # Reduced batch size to avoid OOM errors
        "nT": 10,
        "device": "cuda",
        "window_size": 10,             # seconds (e.g. 1280 samples at 128Hz)
        "eval_datasets": ["WESAD"],    # Not used in dataset loading since data is unified
        "attention_heads": 8,          # New parameter from training config
        "lam1": 100,                   # New parameter from training config
        "lam2": 1,                     # New parameter from training config
        "PATH": "/scratch/bhanu/cuffless_bp/working_weights_trained"  # Updated checkpoint folder
    }

    # Call eval_diffusion once (no loop over dataset names since data is unified)
    tracked_metrics = eval_diffusion(
        window_size=config["window_size"],
        nT=config["nT"],
        batch_size=config["batch_size"],
        device=config["device"],
        PATH=config["PATH"]
    )
    
    # Print BP-specific metrics along with FD
    print(f"\nMAE_SBP: {tracked_metrics['MAE_SBP']:.2f}, MAE_DBP: {tracked_metrics['MAE_DBP']:.2f}")
    print(f"RMSE_SBP: {tracked_metrics['RMSE_SBP']:.2f}, RMSE_DBP: {tracked_metrics['RMSE_DBP']:.2f}")
    print(f"FD: {tracked_metrics['FD']:.2f}")
    
    print("\nPlots have been saved in the ecg_plots directory")
