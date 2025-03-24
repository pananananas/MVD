#!/bin/bash
#SBATCH --job-name=render-objaverse
#SBATCH -A plgtattooai-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH -t 47:59:59
#SBATCH -C memfs
#SBATCH --output=logs/logs_%j.out 

echo "Run started"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export HTTP_TIMEOUT=300

uv run main.py --max_concurrent_downloads=8 --max_download_retries=5 --batch_size=20

echo "Run completed! :>"