#!/bin/bash
#SBATCH --job-name=bp_train_rddm
#SBATCH --partition=gpu_v100_2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
# Load your environment
/scratch/bhanu/cuffless_bp/diffusion/bin/python --version
# Optional: use offline mode if WANDB fails to connect
# export WANDB_MODE=offline
pwd
nvidia-smi
# Navigate to your project directory
cd /scratch/bhanu/cuffless_bp/Cuffless_BP_Estimation_Diffusion_Model

# Run your script
/scratch/bhanu/cuffless_bp/diffusion/bin/python train.py

