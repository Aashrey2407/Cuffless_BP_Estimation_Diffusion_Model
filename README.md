## Environment Setup
```
conda env create -f env.yml && conda activate bpproject
```
try with env.yml and if it doesn't work, use config.yml
**OR**
Create your custom conda environment with 
```
conda create -n [INSERT_ENV_NAME_HERE] python=3.11
```
and run 
```
pip install -r requirements.txt
```

## Waveform Data 
```
 Each recording comprises two records (a waveform record and a matching numerics record) in a single record directory ("folder") with the name of the record. To reduce access time, the record directories have been distributed among ten intermediate-level directories (listed below). The names of these intermediate directories (30, 31, ..., 39) match the first two digits of the record directories they contain.

In almost all cases, the waveform records comprise multiple segments, each of which can be read as a separate record. Each segment contains an uninterrupted recording of a set of simultaneously observed signals, and the signal gains do not change at any time during the segment. Whenever the ICU staff changed the signals being monitored or adjusted the amplitude of a signal being monitored, this event was recorded in the raw data dump, and a new segment begins at that time.

Each composite waveform record includes a list of the segments that comprise it in its master header file. The list begins on the second line of the master header with a layout header file that specifies all of the signals that are observed in any segment belonging to the record. Each segment has its own header file and (except for the layout header) a matching (binary) signal (.dat) file. Occasionally, the monitor may be disconnected entirely for a short time; these intervals are recorded as gaps in the master header file, but there are no header or signal files corresponding to gaps.

The numerics records (designated by the letter n appended to the record name) are not divided into segments, since the storage savings that would be achieved by doing so would be relatively little.

Physiologic waveform records in this database contain up to eight simultaneously recorded signals digitized at 125 Hz with 8-, 10-, or (occasionally) 12-bit resolution. Numerics records typically contain 10 or more time series of vital signs sampled once per second or once per minute.

An example will make this arrangement clear:

    Intermediate directory 31 contains all records with names that begin with 31.
    Record directory 3141595 is contained within intermediate directory 31.
    All files associated with physiologic waveform record 3141595 and its companion numerics record 3141595n are contained within record directory 31/3141595.
        The first line of the master header file for waveform record 314595 (31/3141595/3141595.hea) indicates that the record is 242353557 sample intervals (about 22 days at 125 samples per second) in duration, and that it contains 427 segments and gaps. (See header(5) in the WFDB Applications Guide for details on the format of this text file.) The first segment is named 3141595_0001, and it is 2888500 sample intervals (6 hours, 15 minutes, and 8 seconds, at 125 samples per second) in duration. At the end of the master header file, a comment (# Location: nicu) specifies the ICU in which the recording was made (the neonatal ICU in this case).
        The layout header file for this record (31/3141595/3141595_layout.hea) indicates that five ECG signals (I, II, III, AVR, and "V"), a respiration signal, and a PPG signal are available during portions of the record. (The five ECG signals are not all available simultaneously.)
        The header file for the first segment of this record (31/3141595/3141595_0001.hea) shows that a PPG signal ("PLETH"), a respiration signal, and ECG leads II and AVR are available throughout this initial segment.
    The matching numerics record is named 3141595n, and its header file (31/3141595/3141595n.hea) shows that it is 1938730 sample intervals (about 22 days at 1 sample per second) in duration, and that it contains heart rate (HR, from ECG, as well as PULSE, from one or more pulsatile signals), noninvasive blood pressure (raw as well as systolic, diastolic, and mean), respiration rate, and SpO2.


```
- 3141595_0417 3 125 1913875 16:21:29.296
- Format: 
- [Record name] [Number of Signals] [Sampling Frequency in Hz] [Number of samples] [Timestamp] 
-
-
- 
- 3141595_0417.dat 16 515(254)/pm 10 512 583 -21339 0 RESP
- 3141595_0417.dat 16 1023(0)/NU 10 512 389 -25018 0 PLETH
- 3141595_0417.dat 16 202(410)/mV 10 512 463 11988 0 I

- Here, the format is :
- [signal storing file] [16-bit integers] [ADC gain(ADC zero)/units(mostly pressure in mmHg)] [ADC resolution] [sampling frequency in Hz] [Baseline offset] [Initial value in ADC units] [checksum] [Type of signal]


## RDDM
Here's a comprehensive explanation of the RDDM codebase:

### Repository Overview

This repository implements a Region-Disentangled Diffusion Model (RDDM) for converting PPG signals to ECG signals. The key innovation is that it uses a selective noise addition process that focuses on important regions of interest (ROI) in ECG signals like the QRS complex, rather than adding noise uniformly across the signal.

### Key Files Overview

1. model.py - Contains neural network architectures
2. diffusion.py - Implements the diffusion model logic
3. data.py - Handles data loading and preprocessing 
4. train.py - Training script
5. std_eval.py - Evaluation script
6. metrics.py - Evaluation metrics
7. std_eval.sh - Shell script for running evaluation

### Detailed File Explanations

#### 1. model.py
Contains three main model architectures:

- `DiffusionUNetCrossAttention`: The main U-Net architecture with cross-attention for diffusion
- `ConditionNet`: Network for encoding conditioning information
- `SelfAttention`/`CrossAttentionBlock`: Attention mechanisms for feature processing

Key features:
- Uses a U-Net structure with skip connections
- Incorporates positional encoding for temporal information
- Has both downsampling and upsampling paths
- Uses cross-attention to condition generation on input PPG

#### 2. `diffusion.py`
Implements the core diffusion model logic:

- `ddpm_schedule`: Computes the noise schedule parameters
- `RDDM`: Main diffusion model class that handles:
  - Forward process (adding noise selectively to ROIs)
  - Reverse process (denoising with region disentanglement)
  - Training and sampling procedures

#### 3. data.py
Handles data processing:

- `ECGDataset`: Custom dataset class that:
  - Loads ECG and PPG signals
  - Cleans signals using neurokit2
  - Identifies ROI regions (QRS complexes)
  - Returns aligned ECG-PPG pairs with ROI masks
  
- `get_datasets`: Creates train/test datasets from multiple sources

#### 4. train.py
Training script with:

- Configuration handling
- Data loading
- Model initialization
- Training loop with:
  - Forward/backward passes
  - Loss computation
  - Gradient updates
  - Checkpointing
  - Logging with wandb

#### 5. std_eval.py
Evaluation script that:

- Loads pretrained models
- Runs inference on test data
- Computes metrics:
  - RMSE (Root Mean Square Error)
  - FD (Fréchet Distance)
  - MAE HR (Mean Absolute Error in Heart Rate estimation)

#### 6. metrics.py
Implements evaluation metrics:

- Signal quality metrics (RMSE, FD)
- Clinical metrics (Heart rate estimation)
- Helper functions for signal processing


### Key Aspects to Note

1. **Signal Processing**:
   - Signals need to be at 128Hz sampling rate
   - 4-second windows are used (512 samples)
   - Signals should be normalized to [-1, 1] range

2. **Model Architecture**:
   - Uses two conditioning networks for different aspects
   - Implements region-aware noise addition
   - Uses cross-attention for conditioning

3. **Training Process**:
   - Uses two loss components:
     - DDPM loss for overall signal quality
     - Region loss for ROI preservation
   - Implements curriculum learning through masking

4. **Hardware Requirements**:
   - GPU with CUDA support
   - Sufficient memory for batch processing

Let me explain ECG leads and their significance for PPG-to-ECG translation:

### ECG Leads Explained

### Basic ECG Lead Types

1. **Bipolar Limb Leads**
   - **Lead I**: Between left arm (+) and right arm (-)
   - **Lead II**: Between left leg (+) and right arm (-)
   - **Lead III**: Between left leg (+) and left arm (-)

2. **Augmented Limb Leads**
   - **aVR**: Right arm referenced to LA+LL
   - **aVL**: Left arm referenced to RA+LL
   - **aVF**: Left leg referenced to RA+LA

3. **Chest Leads (Precordial)**
   - **V1-V6**: Six positions across the chest

### Which Lead to Choose?

For PPG-to-ECG translation, **Lead II** is typically the best choice because:

1. It provides the clearest view of:
   - P waves (atrial activity)
   - QRS complex (ventricular activity)
   - T waves (ventricular repolarization)

2. Most PPG-to-ECG research uses Lead II because:
   - It has the highest signal-to-noise ratio
   - Shows the most prominent R peaks
   - Matches the blood flow direction measured by PPG

### Signal Characteristics

| Lead   | Main Features | Best For |
|--------|--------------|-----------|
| Lead II | - Clear P waves<br>- Prominent R peaks<br>- Well-defined T waves | PPG-to-ECG translation |
| Lead I  | - Moderate QRS amplitude<br>- Lateral wall activity | Left-sided heart activity |
| Lead III| - Variable morphology<br>- Often noisy | Inferior wall activity |
| aVR     | - Inverted complex<br>- Right-sided view | Right heart pathology |
| V1-V6   | - Detailed chest view<br>- Complex morphology | Specific heart regions |

When using RDDM, stick with Lead II for:
- Best correlation with PPG signals
- Clearest features for the diffusion model to learn
- Most reliable heart rate estimation
- Compatibility with most existing research