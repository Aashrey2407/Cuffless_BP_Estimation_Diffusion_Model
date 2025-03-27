import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_pradyum import get_datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Add at the beginning of the script, after imports
import os
# Configure PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Optional: Enable memory-efficient attention if using transformer modules
torch.backends.cuda.matmul.allow_tf32 = True

# Create a directory to save plots if it doesn't exist
os.makedirs("ecg_plots_mimic", exist_ok=True)

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

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def plot_ecg_comparison(real_ecg, fake_ecg, sample_idx=0, seconds=5):
    """
    Plot real vs predicted ECG along with vertically flipped versions
    
    Args:
        real_ecg: Ground truth ECG signal
        fake_ecg: Generated ECG signal
        sample_idx: Index of the sample in the batch
        seconds: Number of seconds to plot
    """
    # Calculate number of samples for the time window
    sample_rate = 128  # Hz
    n_samples = int(seconds * sample_rate)
    
    # Extract the first segments
    real_segment = real_ecg[sample_idx, :n_samples]
    fake_segment = fake_ecg[sample_idx, :n_samples]
    
    # Create vertically flipped versions
    real_v_flip = -real_segment
    fake_v_flip = -fake_segment
    
    # Create time axis (in seconds)
    time = np.arange(n_samples) / sample_rate
    
    # Create the plot with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot real ECG
    axs[0].plot(time, real_segment, 'b-', linewidth=1.5)
    axs[0].set_title('Real ECG Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot generated ECG
    axs[1].plot(time, fake_segment, 'r-', linewidth=1.5)
    axs[1].set_title('Generated ECG Signal')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot vertically flipped real ECG
    axs[2].plot(time, real_v_flip, 'b-', linewidth=1.5)
    axs[2].set_title('Vertically Flipped Real ECG Signal')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Plot vertically flipped generated ECG
    axs[3].plot(time, fake_v_flip, 'r-', linewidth=1.5)
    axs[3].set_title('Vertically Flipped Generated ECG Signal')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_xlabel('Time (seconds)')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"ecg_plots_mimic/ecg_comparison_sample_{sample_idx}.png", dpi=300)
    plt.close()
    
    print(f"ECG comparison plot saved to ecg_plots_mimic/ecg_comparison_sample_{sample_idx}.png")

def eval_diffusion(window_size, nT=10, batch_size=8, device="cuda"):

    _, dataset_test = get_datasets("./preprocessed_mimic", window_size)

    # Reduced batch size and workers to prevent OOM errors
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
        nT=nT,
        type="RDDM",
        device="cuda"
    )
    
    # Only use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        dpm = nn.DataParallel(dpm)
        Conditioning_network1 = nn.DataParallel(Conditioning_network1)
        Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    dpm.eval()
    Conditioning_network1.eval()
    Conditioning_network2.eval()

    with torch.no_grad():
        fd_list = []
        fake_ecgs = np.zeros((1, 128*window_size))
        real_ecgs = np.zeros((1, 128*window_size))
        real_ppgs = np.zeros((1, 128*window_size))
        true_rois = np.zeros((1, 128*window_size))
        
        # Flag to indicate if we've plotted the first batch
        first_batch_plotted = False

        for y_ecg, x_ppg, ecg_roi in tqdm(testloader):
            # Clear CUDA cache to prevent OOM
            torch.cuda.empty_cache()
            
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            generated_windows = []

            for ppg_window in torch.split(x_ppg, 128*4, dim=-1):
                if ppg_window.shape[-1] != 128*4:
                    ppg_window = F.pad(ppg_window, (0, 128*4 - ppg_window.shape[-1]), "constant", 0)

                ppg_conditions1 = Conditioning_network1(ppg_window)
                ppg_conditions2 = Conditioning_network2(ppg_window)

                xh = dpm(
                    cond1=ppg_conditions1, 
                    cond2=ppg_conditions2, 
                    mode="sample", 
                    window_size=128*4
                )
                
                generated_windows.append(xh.cpu().numpy())

            xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window_size]

            fd = calculate_FD(y_ecg, torch.from_numpy(xh).to(device))

            # Store data for metrics
            fake_ecgs_batch = xh.reshape(-1, 128*window_size)
            real_ecgs_batch = y_ecg.reshape(-1, 128*window_size).cpu().numpy()
            real_ppgs_batch = x_ppg.reshape(-1, 128*window_size).cpu().numpy()
            true_rois_batch = ecg_roi.reshape(-1, 128*window_size).cpu().numpy()
            
            # Plot the first 3 samples from the first batch
            if not first_batch_plotted:
                for i in range(min(3, len(real_ecgs_batch))):
                    plot_ecg_comparison(
                        real_ecgs_batch, 
                        fake_ecgs_batch, 
                        sample_idx=i,
                        seconds=5  # Plot first 5 seconds
                    )
                first_batch_plotted = True
            
            # Add to the accumulated data
            fake_ecgs = np.concatenate((fake_ecgs, fake_ecgs_batch))
            real_ecgs = np.concatenate((real_ecgs, real_ecgs_batch))
            real_ppgs = np.concatenate((real_ppgs, real_ppgs_batch))
            true_rois = np.concatenate((true_rois, true_rois_batch))
            fd_list.append(fd)

        mae_hr_ecg, rmse_score = evaluation_pipeline(real_ecgs[1:], fake_ecgs[1:])

        tracked_metrics = {
            "RMSE_score": rmse_score,
            "MAE_HR_ECG": mae_hr_ecg,
            "FD": sum(fd_list) / len(fd_list),
        }
        
        # Generate a detailed comparison plot of the entire signals
        plt.figure(figsize=(20, 10))
        
        # Plot mean squared error over time for the first sample
        sample_idx = 0
        mse_over_time = (real_ecgs[1+sample_idx] - fake_ecgs[1+sample_idx])**2
        
        plt.subplot(3, 1, 1)
        plt.plot(real_ecgs[1+sample_idx][:640], 'b-', label='Real ECG')
        plt.title('Real vs Generated ECG (First 5 seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(fake_ecgs[1+sample_idx][:640], 'r-', label='Generated ECG')
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
        plt.savefig("ecg_plots_mimic/ecg_error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Detailed error analysis plot saved to ecg_plots_mimic/ecg_error_analysis.png")

        return tracked_metrics

if __name__ == "__main__":
    # Reduce batch size to avoid CUDA OOM errors
    config = {
        "batch_size": 8,  # Reduced from 32 to 8
        "nT": 10,
        "device": "cuda",
        "window_size": 10, # Seconds
        "eval_datasets": ["WESAD"]
    }

    # TABLE 1 results
    tracked_metrics = eval_diffusion(
        window_size=10,
        nT=10,
        batch_size=8  # Pass the reduced batch size
    )
    print(f"\n: RMSE is {tracked_metrics['RMSE_score']}, FD is {tracked_metrics['FD']}")
    
    # Plot individual beat comparison for deeper analysis
    print("\nPlots have been saved in the ecg_plots_mimic directory")