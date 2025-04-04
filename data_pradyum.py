import torch
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader

'''class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)

def get_datasets(DATA_PATH, window_size):
    # Load ECG and PPG data
    ecg_train = np.load(DATA_PATH + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ppg_train = np.load(DATA_PATH + f"/ppg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ecg_test = np.load(DATA_PATH + f"/ecg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ppg_test = np.load(DATA_PATH + f"/ppg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    
    # Load BP data
    bp_train = np.load(DATA_PATH + f"/bp_train_{window_size}sec.npy", allow_pickle=True)
    bp_test = np.load(DATA_PATH + f"/bp_test_{window_size}sec.npy", allow_pickle=True)
    
    padding_size = 30
    ecg_train = np.pad(ecg_train, ((0, 0), (0, padding_size)), mode='constant')
    ppg_train = np.pad(ppg_train, ((0, 0), (0, padding_size)), mode='constant')
    ecg_test = np.pad(ecg_test, ((0, 0), (0, padding_size)), mode='constant')
    ppg_test = np.pad(ppg_test, ((0, 0), (0, padding_size)), mode='constant')

    ecg_train = np.nan_to_num(ecg_train.astype("float32"))
    ppg_train = np.nan_to_num(ppg_train.astype("float32"))
    ecg_test = np.nan_to_num(ecg_test.astype("float32"))
    ppg_test = np.nan_to_num(ppg_test.astype("float32"))

    # Optionally, apply min-max scaling to ECG and PPG; BP may need separate handling.
    from sklearn.preprocessing import minmax_scale
    dataset_train = ECGBPDataset(
        minmax_scale(ecg_train, (-1, 1), axis=1),
        minmax_scale(ppg_train, (-1, 1), axis=1),
        bp_train.astype("float32")
    )
    dataset_test = ECGBPDataset(
        minmax_scale(ecg_test, (-1, 1), axis=1),
        minmax_scale(ppg_test, (-1, 1), axis=1),
        bp_test.astype("float32")
    )

    return dataset_train, dataset_test'''

class ECGBPDataset(Dataset):
    def __init__(self, ecg_data, ppg_data, bp_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        self.bp_data = bp_data  # bp_data is assumed to have two columns: [SBP, DBP]

    def __getitem__(self, index):
        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        bp = self.bp_data[index]  # Get BP values for this sample
        
        window_size = ecg.shape[-1]

        # Clean the signals using NeuroKit2
        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        ecg = ecg.reshape(1, window_size)
        ecg_roi_array = np.zeros_like(ecg)
        roi_search_window = 20  # number of samples to search before and after R peak

        for r_peak in info["ECG_R_Peaks"]:
            q_start = max(0, r_peak - roi_search_window)
            q_end = r_peak
            q_idx = q_start + np.argmin(ecg[0, q_start:q_end]) if q_end > q_start else r_peak

            s_start = r_peak + 1
            s_end = min(r_peak + roi_search_window, window_size)
            s_idx = s_start + np.argmin(ecg[0, s_start:s_end]) if s_end > s_start else r_peak

            ecg_roi_array[0, q_idx:s_idx+1] = 1

        return ecg.copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy(), bp.copy()

    def __len__(self):
        return len(self.ecg_data)


def get_datasets(DATA_PATH, window_size):
    # Load ECG and PPG data
    ecg_train = np.load(DATA_PATH + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ppg_train = np.load(DATA_PATH + f"/ppg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ecg_test = np.load(DATA_PATH + f"/ecg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)
    ppg_test = np.load(DATA_PATH + f"/ppg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 125*window_size)

    ecg_train = -ecg_train
    ecg_test = -ecg_test
    
    # Load BP data
    bp_train = np.load(DATA_PATH + f"/bp_train_{window_size}sec.npy", allow_pickle=True)
    bp_test = np.load(DATA_PATH + f"/bp_test_{window_size}sec.npy", allow_pickle=True)
    
    padding_size = 30
    ecg_train = np.pad(ecg_train, ((0, 0), (0, padding_size)), mode='constant')
    ppg_train = np.pad(ppg_train, ((0, 0), (0, padding_size)), mode='constant')
    ecg_test = np.pad(ecg_test, ((0, 0), (0, padding_size)), mode='constant')
    ppg_test = np.pad(ppg_test, ((0, 0), (0, padding_size)), mode='constant')

    ecg_train = np.nan_to_num(ecg_train.astype("float32"))
    ppg_train = np.nan_to_num(ppg_train.astype("float32"))
    ecg_test = np.nan_to_num(ecg_test.astype("float32"))
    ppg_test = np.nan_to_num(ppg_test.astype("float32"))

    # Optionally, apply min-max scaling to ECG and PPG; BP may need separate handling.
    from sklearn.preprocessing import minmax_scale
    dataset_train = ECGBPDataset(
        minmax_scale(ecg_train, (-1, 1), axis=1),
        minmax_scale(ppg_train, (-1, 1), axis=1),
        bp_train.astype("float32")
    )
    dataset_test = ECGBPDataset(
        minmax_scale(ecg_test, (-1, 1), axis=1),
        minmax_scale(ppg_test, (-1, 1), axis=1),
        bp_test.astype("float32")
    )

    return dataset_train, dataset_test

