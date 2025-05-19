#!/bin/bash
#SBATCH --job-name=my-mvd
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Run started"

source .venv/bin/activate
# torchrun --nproc_per_node=2 --standalone train.py
python train.py --config config/train_config_all.yaml

echo "Run completed! :>"