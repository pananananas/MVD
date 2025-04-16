#!/bin/bash
#SBATCH --job-name=my-mvd
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Run started"

export MASTER_PORT=12340
export MASTER_ADDR=$(hostname -s)
export WORLD_SIZE=2
export RANK=0

sed -i 's/num_gpus: 1/num_gpus: 2/g' config/train_config.yaml

uv run train.py --config config/train_config.yaml --cuda

echo "Run completed! :>"