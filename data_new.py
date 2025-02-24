import torch
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        # Clean the ECG and PPG signals
        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")

        # Detect R-peaks in the ECG
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)
        r_peaks = info["ECG_R_Peaks"]  # List of detected R-peak indices

        # Compute the first derivative of the ECG signal
        ecg_derivative = np.diff(ecg, n=1, prepend=ecg[0])

        # Create an empty binary ROI mask
        ecg_roi_array = np.zeros(window_size)

        for r_peak in r_peaks:
            # **Find Q-Peak (before R) using zero-crossing in first derivative**
            search_start = max(0, r_peak - 50)  # Search before R
            q_candidates = np.where(np.diff(ecg_derivative[search_start:r_peak]) > 0)[0]  # Zero-crossing (negative to positive)
            q_peak = search_start + (q_candidates[0] if len(q_candidates) > 0 else np.argmin(ecg[search_start:r_peak]))

            # **Find S-Peak (after R) using zero-crossing in first derivative**
            search_end = min(r_peak + 50, window_size)  # Search after R
            s_candidates = np.where(np.diff(ecg_derivative[r_peak:search_end]) > 0)[0]  # Zero-crossing (negative to positive)
            s_peak = r_peak + (s_candidates[0] if len(s_candidates) > 0 else np.argmin(ecg[r_peak:search_end]))

            # **Mark Q-S region as 1 in the mask (QRS complex is the ROI now!)**
            ecg_roi_array[q_peak:s_peak] = 1  

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.reshape(1, window_size).copy()

    def __len__(self):
        return len(self.ecg_data)

def get_datasets(
    DATA_PATH = "../../ingenuity_NAS/21ds94_nas/21ds94_mount/AAAI24/datasets/", 
    datasets=["BIDMC", "CAPNO", "DALIA", "MIMIC-AFib", "WESAD"],
    window_size=4,
    ):

    ecg_train_list = []
    ppg_train_list = []
    ecg_test_list = []
    ppg_test_list = []
    
    for dataset in datasets:

        ecg_train = np.load(DATA_PATH + dataset + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_train = np.load(DATA_PATH + dataset + f"/ppg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        
        ecg_test = np.load(DATA_PATH + dataset + f"/ecg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_test = np.load(DATA_PATH + dataset + f"/ppg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)

        ecg_train_list.append(ecg_train)
        ppg_train_list.append(ppg_train)
        ecg_test_list.append(ecg_test)
        ppg_test_list.append(ppg_test)

    ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
    ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))

    ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
    ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))

    dataset_train = ECGDataset(
        skp.minmax_scale(ecg_train, (-1, 1), axis=1),
        skp.minmax_scale(ppg_train, (-1, 1), axis=1)
    )
    dataset_test = ECGDataset(
        skp.minmax_scale(ecg_test, (-1, 1), axis=1),
        skp.minmax_scale(ppg_test, (-1, 1), axis=1)
    )

    return dataset_train, dataset_test
