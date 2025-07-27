# About the Project

## Core Problem & Solution

- Traditional blood pressure measurement requires cuffs, which are inconvenient and can't provide continuous monitoring
- This solution creates a "cuffless" BP monitor using two common biosignals: ECG (heart electrical activity) and PPG (blood volume changes detected by light)
- The system predicts Arterial Blood Pressure (ABP) waveform, which is then converted to actual BP readings (systolic/diastolic)

## Technical Architecture

- PPG signal → RDDM (Residual Denoising Diffusion Model) → Generated ECG signal
- Generated ECG + Original PPG → LSTM Model → ABP signal → BP values
- The RDDM uses a sophisticated approach with two conditional networks:
  - One specialized for PQRST regions (the main heartbeat pattern in ECG)
  - Another for all other waveform segments
    
## Key Technical Components

- Diffusion Model (RDDM): Based on DDPM (Denoising Diffusion Probabilistic Models) using U-Net architecture for high-quality signal generation
Conditional Networks: Smart preprocessing that treats different parts of the cardiac cycle differently
- LSTM: Handles the temporal relationships between ECG and PPG to predict ABP
- End-to-end Pipeline: PPG → ECG generation → ABP prediction → BP extraction

## Infrastructure & Tools

- High Performance Computing (HPC) cluster with SLURM job scheduling for intensive model training
- Weights & Biases (WandB) for experiment tracking, metrics visualization, and model performance analytics

# Environment Setup
```
conda env create -f env.yml && conda activate bpproject
```
- try with env.yml and if it doesn't work, use config.yml
**OR**
- Create your custom conda environment with 
```
conda create -n [INSERT_ENV_NAME_HERE] python=3.11
```
and run 
```
pip install -r requirements.txt
```

# Waveform Data 
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
```
