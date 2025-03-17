#!/bin/bash
#SBATCH --job-name=render-objaverse
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Downloading GSO started"

uv run download_gso.py \
    -o "GoogleResearch" \
    -c "Scanned Objects by Google Research" \
    -d "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/gso/"

echo "Downloading GSO completed! :>"