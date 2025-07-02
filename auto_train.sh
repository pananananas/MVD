#!/bin/bash

# Configuration
CONFIG_FILE="config/train_config.yaml"
JOB_NAME="train-mvd"
ACCOUNT="plgtattooai-gpu-a100"
PARTITION="plgrid-gpu-a100"
TIME_LIMIT="47:00:00"
MEMORY="1000GB"
OUTPUT_DIR="outputs"
CHECKPOINT_DIR=""
WANDB_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --wandb-id)
            WANDB_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "=== AUTO-TRAIN SCRIPT STARTED ==="
echo "Config: $CONFIG_FILE"
echo "Job started at: $(date)"

# Find the most recent checkpoint if checkpoint dir is provided
RESUME_ARG=""
if [[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]]; then
    # Look for last.ckpt first, then most recent checkpoint
    if [[ -f "$CHECKPOINT_DIR/last.ckpt" ]]; then
        LATEST_CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
        echo "Found last.ckpt: $LATEST_CHECKPOINT"
    else
        LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
        fi
    fi
    
    if [[ -n "$LATEST_CHECKPOINT" ]]; then
        RESUME_ARG="--resume $LATEST_CHECKPOINT"
        echo "Will resume from: $LATEST_CHECKPOINT"
    fi
fi

# Add WandB ID if provided
WANDB_ARG=""
if [[ -n "$WANDB_ID" ]]; then
    WANDB_ARG="--wandb_id $WANDB_ID"
    echo "Using WandB ID: $WANDB_ID"
fi

# Function to resubmit the job
resubmit_job() {
    echo "=== PREPARING TO RESUBMIT JOB ==="
    
    # Get the output directory from the most recent run
    if [[ -z "$CHECKPOINT_DIR" ]]; then
        # Find the most recent output directory (format: YYYY-MM-DD_HH-MM-SS)
        LATEST_OUTPUT_DIR=$(find outputs -maxdepth 1 -type d -name "????-??-??_??-??-??" | sort | tail -1)
        if [[ -n "$LATEST_OUTPUT_DIR" && -d "$LATEST_OUTPUT_DIR/checkpoints" ]]; then
            # Verify that there are actually checkpoint files
            if ls "$LATEST_OUTPUT_DIR/checkpoints"/*.ckpt >/dev/null 2>&1; then
                CHECKPOINT_DIR="$LATEST_OUTPUT_DIR/checkpoints"
                echo "Auto-detected checkpoint directory: $CHECKPOINT_DIR"
            else
                echo "Found output directory but no checkpoint files: $LATEST_OUTPUT_DIR/checkpoints"
            fi
        fi
    fi
    
    # Extract WandB ID from checkpoint if not provided
    if [[ -z "$WANDB_ID" && -n "$CHECKPOINT_DIR" && -f "$CHECKPOINT_DIR/last.ckpt" ]]; then
        echo "Attempting to extract WandB ID from checkpoint..."
        # This will be handled by the Python script itself
    fi
    
    echo "Resubmitting job with:"
    echo "  Config: $CONFIG_FILE"
    echo "  Checkpoint dir: $CHECKPOINT_DIR"
    echo "  WandB ID: $WANDB_ID"
    
    # Submit the new job
    srun --job-name=$JOB_NAME \
         -A $ACCOUNT \
         -p $PARTITION \
         --nodes=1 \
         --cpus-per-task=16 \
         --gres=gpu:4 \
         --ntasks-per-node=4 \
         --mem=$MEMORY \
         -t $TIME_LIMIT \
         -C memfs \
         bash auto_train.sh --config "$CONFIG_FILE" \
         $([ -n "$CHECKPOINT_DIR" ] && echo "--checkpoint-dir $CHECKPOINT_DIR") \
         $([ -n "$WANDB_ID" ] && echo "--wandb-id $WANDB_ID") &
    
    echo "New job submitted! PID: $!"
    echo "Current job will exit gracefully."
}

# Set up signal handlers for graceful shutdown
trap 'echo "Received termination signal, cleaning up..."; resubmit_job; exit 0' SIGTERM SIGINT

# Calculate when to resubmit (45 hours = 162000 seconds, leaving 2 hours buffer)
RESUBMIT_TIME=162000

# Start the resubmission timer in background
(
    sleep $RESUBMIT_TIME
    echo "Time limit approaching, triggering resubmission..."
    resubmit_job
    # Send signal to main process
    kill -TERM $$
) &
TIMER_PID=$!

echo "Started resubmission timer (PID: $TIMER_PID) for $RESUBMIT_TIME seconds"

# Activate virtual environment and run training
source .venv/bin/activate

echo "=== STARTING TRAINING ==="
echo "Command: python train.py --config $CONFIG_FILE $RESUME_ARG $WANDB_ARG"

# Run the training with proper error handling
python train.py --config "$CONFIG_FILE" $RESUME_ARG $WANDB_ARG

TRAIN_EXIT_CODE=$?

# Kill the timer if training finished normally
kill $TIMER_PID 2>/dev/null

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "=== TRAINING COMPLETED SUCCESSFULLY ==="
    echo "Training finished at: $(date)"
else
    echo "=== TRAINING FAILED ==="
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Training failed at: $(date)"
    # Still resubmit to retry
    resubmit_job
fi 