#!/bin/bash
#SBATCH --job-name=difftrain_rddm
#SBATCH --partition=gpu_a100_8
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
# Load your environment
spack load python@3.9.13%gcc@8.5.0
python --version
# source /scratch/bhanu/cuffless_bp/diffu/bin/activate
python --version
# Optional: use offline mode if WANDB fails to connect
# export WANDB_MODE=offline
pwd
nvidia-smi
# Navigate to your project directory
cd /home/ssdan/cuffless_BP_prediction/aashrey/Cuffless_BP_Estimation_Diffusion_Model/

ulimit -n 2048
# Run your script
python train.py
