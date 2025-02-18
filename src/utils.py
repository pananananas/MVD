import os
import torch
import time
from pathlib import Path
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, pipeline, optimizer, epoch, loss, metrics=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': pipeline.unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)
            
        return checkpoint_path
    
    def load_latest(self, pipeline, optimizer):
        if not self.checkpoints:
            return None
            
        latest_checkpoint = self.checkpoints[-1]
        checkpoint = torch.load(latest_checkpoint)
        
        pipeline.unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


def create_output_dirs(base_dir):
    """Create necessary output directories for training"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    
    dirs = {
        'checkpoints': run_dir / 'checkpoints',
        'samples': run_dir / 'samples',
        'logs': run_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs 