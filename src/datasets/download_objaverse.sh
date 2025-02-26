#!/bin/bash
#SBATCH --job-name=download-objaverse
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Run started"

# Set up temporary directory
export TMPDIR="/net/pr2/projects/plgrid/plggtattooai/tmp"
mkdir -p $TMPDIR
echo "Using temporary directory: $TMPDIR"

uv run src/datasets/download_objaverse.py

echo "Run completed! :>"