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

    def get_datasets(DATA_PATH,window_size):
        ecg_train = np.load(DATA_PATH + f"/ecg_train_{window_size}sec.npy",allow_pickle=True).reshape(-1,125*window_size)
        ppg_train = np.load(DATA_PATH + f"/ppg_train_{window_size}sec.npy",allow_pickle=True).reshape(-1,125*window_size)
        ecg_test = np.load(DATA_PATH + f"/ecg_train_{window_size}sec.npy",allow_pickle=True).reshape(-1,125*window_size)
        ppg_test = np.load(DATA_PATH + f"/ppg_test_{window_size}sec.npy",allow_pickle=True).reshape(-1,125*window_size)

        ecg_train = np.nan_to_num(ecg_train.astype("float32"))
        ppg_train = np.nan_to_num(ppg_train.astype("float32"))
        ecg_test = np.nan_to_num(ecg_test.astype("float32"))
        ppg_test = np.nan_to_num(ppg_test.astype("float32"))

        dataset_train = ECGDataset(skp.minmax_scale(ecg_train,(-1,1),axis = 1),skp.minmax_scale(ppg_train,(-1,1),axis=1))
        dataset_test = ECGDataset(skp.minmax_scale(ecg_test,(-1,1),axis=1),skp.minmax_scale(ppg_test,(-1,1),axis=1))

        return dataset_train,dataset_test
