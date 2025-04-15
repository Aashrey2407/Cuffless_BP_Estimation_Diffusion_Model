import duckdb
import numpy as np
import torch
from torch.utils.data import Dataset
import neurokit2 as nk
from sklearn.preprocessing import minmax_scale

class ECGBPDataset(Dataset):
    """
    Custom dataset that returns:
      - ECG signal (cleaned and reshaped to [1, window_length])
      - PPG signal (cleaned and reshaped to [1, window_length])
      - ECG ROI array (binary mask computed from detected R-peaks)
      - BP values as a two-element array: [SegSBP, SegDBP]
    """
    def __init__(self, ecg_data, ppg_data, bp_data):
        self.ecg_data = ecg_data  # Expected shape: (N, L)
        self.ppg_data = ppg_data  # Expected shape: (N, L)
        self.bp_data = bp_data    # Expected shape: (N, 2)

    def __getitem__(self, index):
        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        bp  = self.bp_data[index]  # [SBP, DBP]

        window_size = ecg.shape[-1]

        # Clean the signals using NeuroKit2
        ppg_cleaned = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg_cleaned = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=128, method="pantompkins1985",
                                correct_artifacts=True, show=False)

        # Reshape ECG for U-Net input (channel-first)
        ecg_cleaned = ecg_cleaned.reshape(1, window_size)
        # Create an ROI mask based on detected R-peaks
        ecg_roi_array = np.zeros_like(ecg_cleaned)
        roi_window = 20  # samples before and after the R-peak

        for r_peak in info["ECG_R_Peaks"]:
            q_start = max(0, r_peak - roi_window)
            q_end = r_peak
            # Find the valley before the R-peak
            q_idx = q_start + np.argmin(ecg_cleaned[0, q_start:q_end]) if (q_end > q_start) else r_peak
            s_start = r_peak + 1
            s_end = min(r_peak + roi_window, window_size)
            # Find the valley after the R-peak
            s_idx = s_start + np.argmin(ecg_cleaned[0, s_start:s_end]) if (s_end > s_start) else r_peak
            # Mark the region between the valleys as ROI (set to 1)
            ecg_roi_array[0, q_idx:s_idx+1] = 1

        return ecg_cleaned.copy(), ppg_cleaned.reshape(1, window_size).copy(), ecg_roi_array.copy(), bp.copy()

    def __len__(self):
        return len(self.ecg_data)

def get_dataset_from_parquet(parquet_path, window_size):
    """
    Loads the dataset from the provided Parquet file.
    Assumes the Parquet file contains at least the following columns:
      - ECG_F: Filtered ECG signal (DOUBLE[])
      - PPG_F: Filtered PPG signal (DOUBLE[])
      - SegSBP: Segmented systolic BP (DOUBLE[] or scalar)
      - SegDBP: Segmented diastolic BP (DOUBLE[] or scalar)
    
    Converts the columns to NumPy arrays, applies padding and min-max scaling, and returns
    an ECGBPDataset instance.
    
    :param parquet_path: Path to the Parquet file (e.g., "dataset/Train_Processed.parquet")
    :param window_size: Duration in seconds (for compatibility; signals are stored as arrays)
    :return: ECGBPDataset instance
    """
    query = f"""
        SELECT
            ECG_F,
            PPG_F,
            SegSBP,
            SegDBP
        FROM '{parquet_path}'
    """
    print(f"📦 Executing DuckDB query on: {parquet_path}")
    df = duckdb.query(query).to_df()
    print("✅ DuckDB query complete.")

    # Convert columns containing signal data into NumPy arrays.
    ecg_list = df['ECG_F'].tolist()
    ppg_list = df['PPG_F'].tolist()
    ecg_arr = np.array(ecg_list)  # Shape: (N, L)
    ppg_arr = np.array(ppg_list)  # Shape: (N, L)

    # Convert BP values. They may be stored as scalar values or single-element arrays.
    sbp_list = df['SegSBP'].tolist()
    dbp_list = df['SegDBP'].tolist()
    sbp_arr = np.array(sbp_list)
    dbp_arr = np.array(dbp_list)
    if len(sbp_arr.shape) == 1:
        sbp_arr = sbp_arr.reshape(-1, 1)
    if len(dbp_arr.shape) == 1:
        dbp_arr = dbp_arr.reshape(-1, 1)
    bp_arr = np.concatenate([sbp_arr, dbp_arr], axis=1)  # Shape: (N, 2)

    # Optionally pad the signals (adds 30 zeros at the end).
    padding_size = 30
    ecg_arr = np.pad(ecg_arr, ((0, 0), (0, padding_size)), mode='constant')
    ppg_arr = np.pad(ppg_arr, ((0, 0), (0, padding_size)), mode='constant')

    # Convert to float32 and scale to the range [-1, 1]
    ecg_arr = np.nan_to_num(ecg_arr.astype("float32"))
    ppg_arr = np.nan_to_num(ppg_arr.astype("float32"))
    ecg_arr = minmax_scale(ecg_arr, feature_range=(-1, 1), axis=1)
    ppg_arr = minmax_scale(ppg_arr, feature_range=(-1, 1), axis=1)

    dataset = ECGBPDataset(ecg_arr, ppg_arr, bp_arr)
    return dataset

def get_datasets(window_size):
    """
    Loads both training and test datasets from their respective Parquet files
    and returns them as ECGBPDataset instances.

    :param train_parquet_path: Path to the training dataset Parquet file.
    :param test_parquet_path: Path to the test dataset Parquet file.
    :param window_size: Duration in seconds (for compatibility).
    :return: (train_dataset, test_dataset)
    """
    train_parquet_path = "/scratch/ssdan/Cuffless-Blood-Pressure/dataset/Train_Processed.parquet"
    test_parquet_path  = "/scratch/ssdan/Cuffless-Blood-Pressure/dataset/CalFree_Test_Processed.parquet"
    print("🟡 Loading training dataset...")
    train_dataset = get_dataset_from_parquet(train_parquet_path, window_size)
    print("✅ Training dataset loaded. Total samples:", len(train_dataset))

    print("🟡 Loading test dataset...")
    test_dataset = get_dataset_from_parquet(test_parquet_path, window_size)
    print("✅ Test dataset loaded. Total samples:", len(test_dataset))

    return train_dataset, test_dataset

# -------------------------------
# Example usage (for testing)
# -------------------------------
if __name__ == "__main__":
    import time
    start = time.time()

    # Define the Parquet file paths:
    
    window_size = 10  # Adjust if needed

    # Load both datasets:
    train_dataset, test_dataset = get_datasets(train_parquet, test_parquet, window_size)
    
    # Print sample details from both datasets:
    sample_train = train_dataset[0]
    sample_test  = test_dataset[0]
    print("\nSample from training dataset:")
    print("    ECG shape       :", sample_train[0].shape)
    print("    PPG shape       :", sample_train[1].shape)
    print("    ECG ROI shape   :", sample_train[2].shape)
    print("    BP (SBP, DBP)   :", sample_train[3])
    
    print("\nSample from test dataset:")
    print("    ECG shape       :", sample_test[0].shape)
    print("    PPG shape       :", sample_test[1].shape)
    print("    ECG ROI shape   :", sample_test[2].shape)
    print("    BP (SBP, DBP)   :", sample_test[3])
    
    end = time.time()
    print(f"\n⏱️ Total time taken: {end - start:.2f} seconds")
