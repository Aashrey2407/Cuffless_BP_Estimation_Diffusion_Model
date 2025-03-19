import os
import wfdb
import numpy as np
import torch
import sklearn.preprocessing as skp
import neurokit2 as nk
from torch.utils.data import Dataset, DataLoader

def segment_signal(signal, window_size):
    """
    Break a 1D signal into non-overlapping segments of length window_size.
    """
    n_samples = len(signal)
    n_segments = n_samples // window_size
    signal = signal[:n_segments * window_size]  # Trim extra samples
    return signal.reshape(n_segments, window_size)

class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):
        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        window_size = ecg.shape[-1]

        # Clean the signals using a sampling rate of 125 Hz
        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=125)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=125, method="pantompkins1985")

        # Detect R-peaks in the ECG signal
        _, info = nk.ecg_peaks(ecg, sampling_rate=125, method="pantompkins1985", correct_artifacts=True, show=False)
        r_peaks = info["ECG_R_Peaks"]

        # Compute the first derivative of the ECG signal
        ecg_derivative = np.diff(ecg, n=1, prepend=ecg[0])

        # Create a binary ROI mask for the QRS complex
        ecg_roi_array = np.zeros(window_size)
        for r_peak in r_peaks:
            search_start = max(0, r_peak - 50)
            q_candidates = np.where(np.diff(ecg_derivative[search_start:r_peak]) > 0)[0]
            q_peak = search_start + (q_candidates[0] if len(q_candidates) > 0 else np.argmin(ecg[search_start:r_peak]))
            search_end = min(r_peak + 50, window_size)
            s_candidates = np.where(np.diff(ecg_derivative[r_peak:search_end]) > 0)[0]
            s_peak = r_peak + (s_candidates[0] if len(s_candidates) > 0 else np.argmin(ecg[r_peak:search_end]))
            ecg_roi_array[q_peak:s_peak] = 1

        return (
            ecg.reshape(1, window_size).copy(),
            ppg.reshape(1, window_size).copy(),
            ecg_roi_array.reshape(1, window_size).copy()
        )

    def __len__(self):
        return len(self.ecg_data)

def read_physionet_record(record_dir, record_name, window_size=500):
    """
    Reads a PhysioNet record (without file extensions) from record_dir.
    Assumes that the .dat and .hea files are present.
    Returns segmented ECG (from channel 'II') and PPG (from channel 'PLETH') signals.
    """
    record_path = os.path.join(record_dir, record_name)
    record = wfdb.rdrecord(record_path)
    sig_names = record.sig_name

    try:
        idx_ppg = sig_names.index("PLETH")
    except ValueError:
        idx_ppg = None
    try:
        idx_ecg = sig_names.index("II")
    except ValueError:
        idx_ecg = None

    ecg = record.p_signal[:, idx_ecg] if idx_ecg is not None else None
    ppg = record.p_signal[:, idx_ppg] if idx_ppg is not None else None

    # Flatten and segment the signals into 4-second chunks (500 samples)
    if ecg is not None:
        ecg = ecg.flatten()
        ecg = segment_signal(ecg, window_size)
    if ppg is not None:
        ppg = ppg.flatten()
        ppg = segment_signal(ppg, window_size)
    return ecg, ppg

def get_datasets_physionet(data_dir, record_list, window_size=500):
    """
    Reads PhysioNet records from data_dir given a list of record names (without extensions)
    and returns a single ECGDataset instance with segmented data.
    """
    ecg_segments = []
    ppg_segments = []
    for record_name in record_list:
        ecg, ppg = read_physionet_record(data_dir, record_name, window_size)
        if ecg is not None and ppg is not None:
            ecg_segments.append(ecg)
            ppg_segments.append(ppg)
    # Concatenate segments from all records along axis 0
    ecg_data = np.concatenate(ecg_segments, axis=0)
    ppg_data = np.concatenate(ppg_segments, axis=0)
    # Scale signals to range [-1, 1]
    ecg_data = skp.minmax_scale(ecg_data, (-1, 1), axis=1)
    ppg_data = skp.minmax_scale(ppg_data, (-1, 1), axis=1)
    dataset = ECGDataset(ecg_data, ppg_data)
    return dataset

# Example usage:
# data_dir = "/path/to/physionet/records"
# record_list = ["3141595_0001"]  # List of record names (without file extensions)
# dataset = get_datasets_physionet(data_dir, record_list)
