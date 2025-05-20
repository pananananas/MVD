#!/bin/bash
#SBATCH --job-name=filter-objaverse
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Filtering script started"

uv run src/data/cleaning/filter_by_contrast.py

echo "Run completed! :>"