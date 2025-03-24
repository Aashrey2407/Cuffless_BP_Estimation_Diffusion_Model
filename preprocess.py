import wfdb
import numpy as np
from tqdm import tqdm
import os
import sklearn.preprocessing as skp

def load_and_preprocess_record(record_path, window_size=4):
    """
    Load and preprocess a MIMIC-II record
    Args:
        record_path: Path to the record without extension
        window_size: Window size in seconds (default 4)
    Returns:
        ppg_windows, ecg_windows: Preprocessed signal windows
    """
    # Read the record
    record = wfdb.rdrecord(record_path)
    fs = record.fs  # Usually 125Hz
    
    # Find PPG and ECG channels
    ppg_idx = None
    ecg_idx = None
    
    for idx, name in enumerate(record.sig_name):
        if 'PLETH' in name.upper() or 'PPG' in name.upper():
            ppg_idx = idx
        # Look for Lead II first, if not found, use other leads
        elif 'II' in name.upper():
            ecg_idx = idx
        elif ecg_idx is None and ('I' in name.upper() or 'III' in name.upper() 
                                or 'AVR' in name.upper() or 'V' in name.upper()):
            ecg_idx = idx
    
    if ppg_idx is None or ecg_idx is None:
        raise ValueError(f"Could not find PPG or ECG in channels: {record.sig_name}")
    
    # Extract signals
    ppg = record.p_signal[:, ppg_idx]
    ecg = record.p_signal[:, ecg_idx]
    
    # Resample to 128Hz if needed
    if fs != 128:
        from scipy.signal import resample
        n_samples = int(len(ppg) * 128 / fs)
        ppg = resample(ppg, n_samples)
        ecg = resample(ecg, n_samples)
    
    # Split into windows
    samples_per_window = window_size * 128
    n_windows = len(ppg) // samples_per_window
    
    ppg_windows = ppg[:n_windows * samples_per_window].reshape(-1, samples_per_window)
    ecg_windows = ecg[:n_windows * samples_per_window].reshape(-1, samples_per_window)
    
    return ppg_windows, ecg_windows

def create_dataset(record_paths, output_dir, window_size=4, train_split=0.8):
    """Process multiple records and create train/test datasets"""
    all_ppg = []
    all_ecg = []
    
    for record_path in tqdm(record_paths):
        try:
            ppg_windows, ecg_windows = load_and_preprocess_record(record_path, window_size)
            all_ppg.append(ppg_windows)
            all_ecg.append(ecg_windows)
        except Exception as e:
            print(f"Error processing {record_path}: {str(e)}")
            continue
    
    # Concatenate all windows
    all_ppg = np.concatenate(all_ppg)
    all_ecg = np.concatenate(all_ecg)
    
    # Split into train/test
    n_train = int(len(all_ppg) * train_split)
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/ppg_train_{window_size}sec.npy", all_ppg[:n_train])
    np.save(f"{output_dir}/ppg_test_{window_size}sec.npy", all_ppg[n_train:])
    np.save(f"{output_dir}/ecg_train_{window_size}sec.npy", all_ecg[:n_train])
    np.save(f"{output_dir}/ecg_test_{window_size}sec.npy", all_ecg[n_train:])

# Usage example
if __name__ == "__main__":
    # List your .dat files (without extension)
    record_paths = [
        "path/to/3141595_0001",
        "path/to/3141595_0002",
    ]
    
    output_dir = "preprocessed_mimic"
    create_dataset(record_paths, output_dir)