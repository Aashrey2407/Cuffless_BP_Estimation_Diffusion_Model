import torch
import numpy as np
import torch.nn as nn
from statistics import mean
from math import floor
from functools import reduce
from typing import Dict, Tuple
import neurokit2 as nk
import torch.nn.functional as F
from model import DiffusionUNetCrossAttention, ConditionNet

def ddpm_schedule(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedule for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "beta_t": beta_t
    }

class NaiveDDPM(nn.Module):
    def __init__(
        self,
        eps_model,
        betas,
        n_T,
        criterion = nn.MSELoss(),
    ):
        super(NaiveDDPM, self).__init__()
        self.eps_model = eps_model
        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def forward(self, x=None, cond=None, mode="train", window_size=128*4):

        if mode == "train":
            
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
                x.device
            ) 

            eps = torch.randn_like(x)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )  

            return self.criterion(eps, self.eps_model(x_t, cond, _ts / self.n_T))

        elif mode == "sample":

            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                
                z = torch.randn(n_sample, 1, window_size).to(device) if i > 1 else 0

                eps = self.eps_model(x_i, cond, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i

class RDDM(nn.Module):
    def __init__(
        self,
        eps_model,
        region_model,
        betas,
        n_T,
        criterion = nn.MSELoss(),
    ):
        super(RDDM, self).__init__()
        self.eps_model = eps_model
        self.region_model = region_model

        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def create_noise_in_regions(self, patch_labels):

        patch_roi = torch.round(patch_labels)

        mask = patch_roi == 1
        random_noise = torch.randn_like(patch_roi)
        masked_noise = random_noise * mask.float()
        
        return masked_noise, random_noise

    def forward(self, x=None, cond1=None, cond2=None, mode="train", patch_labels=None, window_size=128*4):

        if mode == "train":
            
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
                x.device
            ) 

            eps, unmasked_eps = self.create_noise_in_regions(patch_labels)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )  

            x_t_unmasked = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * unmasked_eps
            )

            pred_x_t = self.region_model(x_t_unmasked, cond2, _ts / self.n_T)

            pred_masked_eps = self.eps_model(x_t, cond1, _ts / self.n_T)

            ddpm_loss = self.criterion(eps, pred_masked_eps)

            region_loss = self.criterion(pred_x_t, x_t)

            return ddpm_loss, region_loss

        elif mode == "sample":

            n_sample = cond1["down_conditions"][-1].shape[0]
            device = cond1["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                
                if i > 1:
                    z = torch.randn(n_sample, 1, window_size).to(device)

                else:
                    z = 0
            
                # rho_phi estimates the trajectory from Gaussian manifold to Masked Gaussian manifold
                x_i = self.region_model(x_i, cond2, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                # epsilon_theta predicts the noise that needs to be removed to move from Masked Gaussian manifold to ECG manifold
                eps = self.eps_model(x_i, cond1, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i

class BP_Diffusion(nn.Module):
    def __init__(
        self,
        eps_model,
        region_model,
        betas,
        n_T,
        criterion=nn.MSELoss(),
    ):
        super(BP_Diffusion, self).__init__()
        self.eps_model = eps_model
        self.region_model = region_model

        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps with device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def create_noise_in_regions(self, patch_labels):
        patch_roi = torch.round(patch_labels)
        mask = patch_roi == 1
        random_noise = torch.randn_like(patch_roi)
        masked_noise = random_noise * mask.float()
        return masked_noise, random_noise

    def forward(self, x=None, cond1=None, cond2=None, mode="train", patch_labels=None, window_size=128*4):
        if mode == "train":
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)

            eps, unmasked_eps = self.create_noise_in_regions(patch_labels)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )
            x_t_unmasked = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * unmasked_eps
            )

            pred_x_t = self.region_model(x_t_unmasked, cond2, _ts / self.n_T)
            pred_masked_eps = self.eps_model(x_t, cond1, _ts / self.n_T)

            ddpm_loss = self.criterion(eps, pred_masked_eps)
            region_loss = self.criterion(pred_x_t, x_t)

            # === New: Compute Alignment Losses ===
            # We'll use the clean ECG (x) as the ground truth and the output of region_model (pred_x_t) as the predicted ECG.
            # First, detach and convert to NumPy arrays for peak detection.
            # We assume signals are of shape [batch, 1, window_size].
            true_ecg_np = x.squeeze(1).detach().cpu().numpy()
            pred_ecg_np = pred_x_t.squeeze(1).detach().cpu().numpy()

            L_position_list = []
            L_amplitude_list = []
            L_freq_list = []
            # Loop over each sample in the batch
            for i in range(true_ecg_np.shape[0]):
                # Detect R-peaks in both true and predicted ECG using NeuroKit2
                _, info_true = nk.ecg_peaks(true_ecg_np[i], sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)
                _, info_pred = nk.ecg_peaks(pred_ecg_np[i], sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)
                r_true = np.array(info_true["ECG_R_Peaks"])
                r_pred = np.array(info_pred["ECG_R_Peaks"])
                # For L_position, match the first n_min peaks from each signal.
                n_min = min(len(r_true), len(r_pred))
                if n_min > 0:
                    L_position_sample = np.mean(np.abs(r_true[:n_min] - r_pred[:n_min]))
                else:
                    L_position_sample = 0.0
                # For L_amplitude, compare the maximum amplitude.
                L_amplitude_sample = np.abs(np.max(pred_ecg_np[i]) - np.max(true_ecg_np[i]))
                # For L_freq, compare the number of detected R-peaks.
                L_freq_sample = np.abs(len(r_pred) - len(r_true))
                
                L_position_list.append(L_position_sample)
                L_amplitude_list.append(L_amplitude_sample)
                L_freq_list.append(L_freq_sample)

            # Average the losses over the batch and convert to torch tensors.
            L_position = torch.tensor(np.mean(L_position_list), device=x.device, dtype=torch.float)
            L_amplitude = torch.tensor(np.mean(L_amplitude_list), device=x.device, dtype=torch.float)
            L_freq = torch.tensor(np.mean(L_freq_list), device=x.device, dtype=torch.float)
            
            L_scale = L_position + L_amplitude
            alignment_loss = L_scale + L_freq

            generated_ecg = self.forward(x=x, cond1=cond1, cond2=cond2, mode="sample", patch_labels=patch_labels, window_size=window_size)
            return ddpm_loss, region_loss, alignment_loss, generated_ecg

        elif mode == "sample":
            n_sample = cond1["down_conditions"][-1].shape[0]
            device = cond1["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                if i > 1:
                    z = torch.randn(n_sample, 1, window_size).to(device)
                else:
                    z = 0

                x_i = self.region_model(x_i, cond2, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
                eps = self.eps_model(x_i, cond1, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            return x_i

'''
def compute_alignment_loss(generated_ecg, real_ecg, sampling_rate=128):
    """
    Computes L_a = L_scale + L_freq, where:
      L_scale = L_position + L_amplitude
      L_position = average distance between R-peaks
      L_amplitude = difference in amplitude (max-min)
      L_freq = |N_g - N_t| (difference in # of R-peaks or BPM)
    """
    # 1) R-peak detection for real ECG
    _, info_real = nk.ecg_peaks(real_ecg, sampling_rate=sampling_rate, method="pantompkins1985")
    real_peaks = info_real["ECG_R_Peaks"]

    # 2) R-peak detection for generated ECG
    _, info_gen = nk.ecg_peaks(generated_ecg, sampling_rate=sampling_rate, method="pantompkins1985")
    gen_peaks = info_gen["ECG_R_Peaks"]

    # 3) L_position (assume same length for both signals)
    N_min = min(len(real_peaks), len(gen_peaks))
    if N_min > 0:
        p_real = np.array(real_peaks[:N_min])
        p_gen = np.array(gen_peaks[:N_min])
        L_position = np.mean(np.abs(p_real - p_gen))
    else:
        L_position = 0.0

    # 4) L_amplitude
    E_t = real_ecg.max() - real_ecg.min()  # amplitude of real
    E_g = generated_ecg.max() - generated_ecg.min()  # amplitude of generated
    L_amplitude = abs(E_g - E_t)

    # 5) L_freq
    # heart rate difference ~ # of R-peaks difference
    L_freq = abs(len(gen_peaks) - len(real_peaks))

    # Combine them
    L_scale = L_position + L_amplitude
    L_a = L_scale + L_freq
    return L_a


class BPDiffusion(nn.Module):
    def __init__(
        self,
        eps_model,             # model that predicts noise
        region_model=None,     # optional second model if needed for x_t^{[p]}
        betas=(1e-4, 0.2),
        n_T=1000,
        lam1=100,              # λ1
        lam2=1,                # λ2
        criterion=nn.MSELoss(),
        sampling_rate=128
    ):
        super(BPDiffusion, self).__init__()
        self.eps_model = eps_model
        self.region_model = region_model  # optional if you want a separate region-based model
        self.n_T = n_T
        self.lam1 = lam1
        self.lam2 = lam2
        self.criterion = criterion
        self.sampling_rate = sampling_rate  # for alignment losses

        # Register buffers for the diffusion schedule
        schedule = ddpm_schedule(betas[0], betas[1], n_T)
        for k, v in schedule.items():
            self.register_buffer(k, v)

    def forward(self, x=None, cond=None, mode="train", patch_labels=None):
        """
        x: (batch, channels=1, length) - real ECG
        cond: dictionary of conditioning features
        patch_labels: QRS mask (binary)
        mode: 'train' or 'sample'
        """
        if mode == "train":
            # 1) Sample a random time step
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  # shape: (batch,)

            # 2) Generate Gaussian noise
            eps = torch.randn_like(x)  # same shape as x

            # 3) Create QRS mask and non-QRS mask
            qrs_mask = torch.round(patch_labels)  # shape: same as x
            non_qrs_mask = 1.0 - qrs_mask

            # PHASE 1: (0 ≤ t < T/2) Noise in QRS region
            half_T = self.n_T // 2
            x_half = (
                self.sqrtab[half_T, None, None] * x
                + self.sqrtmab[half_T, None, None] * (qrs_mask * eps)
            )

            # PHASE 2: (T/2 ≤ t ≤ T) Noise in non-QRS region
            x_t = (
                self.sqrtab[_ts, None, None] * x_half
                + self.sqrtmab[_ts, None, None] * (non_qrs_mask * eps)
            )

            # 4) Noise Prediction (first part of L_q)
            pred_noise = self.eps_model(x_t, cond, _ts / self.n_T)  # shape: same as x
            # Compare predicted noise in QRS region with actual noise in QRS
            l_q_part1 = self.criterion(qrs_mask * eps, qrs_mask * pred_noise)  # MSE in QRS region

            # 5) If region_model is used to get x_t^{[p]}, do so:
            #    (this is optional if you want a second model to reconstruct or partial denoise)
            if self.region_model is not None:
                x_t_pred = self.region_model(x_t, cond, _ts / self.n_T)  # x_t^{[p]}
                # second part: (x_T - x_t) - x_t^{[p]} => we approximate x_T ~ x (the real ECG)
                l_q_part2 = self.criterion((x - x_t), x_t_pred)
            else:
                # if no region_model, fallback to 0 for second part
                l_q_part2 = 0.0

            # Full L_q from the paper:
            L_q = self.lam1 * l_q_part1 + self.lam2 * l_q_part2

            # 6) Alignment Loss (L_a)
            # We can treat x_t as a "generated" ECG sample. Or, you might want a separate reverse pass.
            # For demonstration, let's compute alignment vs. real ECG:
            # Convert x, x_t to CPU numpy for neurokit2
            # shape: (batch, 1, length)
            # We'll do a batch average for alignment
            L_a = 0.0
            x_t_np = x_t.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()

            batch_size = x.shape[0]
            for b in range(batch_size):
                gen_ecg = x_t_np[b, 0, :]  # single ECG
                real_ecg = x_np[b, 0, :]
                L_a_single = compute_alignment_loss(gen_ecg, real_ecg, sampling_rate=self.sampling_rate)
                L_a += L_a_single
            L_a = L_a / batch_size  # average alignment loss over batch

            # Final Loss: L_q + L_a
            total_loss = L_q + L_a

            return total_loss, L_q, L_a

        elif mode == "sample":
            # Reverse diffusion sampling from noise to ECG
            # This code is standard DDPM sampling; you could adapt for 2-phase if desired
            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            window_size = x.shape[-1] if x is not None else 512

            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, 1, window_size).to(device) if i > 1 else 0
                pred_noise = self.eps_model(x_i, cond, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                x_i = (
                    self.oneover_sqrta[i] * (x_i - pred_noise * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i

'''



def freeze_model(model):

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    
    return model

def load_pretrained_DPM(PATH, nT, type="RDDM", device="cuda"):

    if type == "RDDM":

        dpm = RDDM(
            eps_model=DiffusionUNetCrossAttention(512, 1, device),
            region_model=DiffusionUNetCrossAttention(512, 1, device),
            betas=(1e-4, 0.2), 
            n_T=nT
        ).to(device)
        
        dpm.load_state_dict(torch.load(PATH + "rddm_main_network.pth"))

        dpm = freeze_model(dpm)

        Conditioning_network1 = ConditionNet().to(device)
        Conditioning_network1.load_state_dict(torch.load(PATH + "rddm_condition_encoder_1.pth"))
        Conditioning_network1 = freeze_model(Conditioning_network1)

        Conditioning_network2 = ConditionNet().to(device)
        Conditioning_network2.load_state_dict(torch.load(PATH + "rddm_condition_encoder_2.pth"))
        Conditioning_network2 = freeze_model(Conditioning_network2)
 
        return dpm, Conditioning_network1, Conditioning_network2
    
    else: # Naive DDPM

        dpm = NaiveDDPM(
            eps_model=DiffusionUNetCrossAttention(512, 1, device),
            betas=(1e-4, 0.2), 
            n_T=nT
        ).to(device)

        dpm.load_state_dict(torch.load(PATH + f"ddpm_main_network_{nT}.pth"))
        dpm = freeze_model(dpm)

        Conditioning_network = ConditionNet().to(device)
        Conditioning_network.load_state_dict(torch.load(PATH + f"ddpm_condition_encoder_{nT}.pth"))
        Conditioning_network = freeze_model(Conditioning_network)
        
        return dpm, Conditioning_network, None