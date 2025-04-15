from datetime import datetime
from pathlib import Path
import time
import os


def create_output_dirs(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    
    dirs = {
        'checkpoints': run_dir / 'checkpoints',
        'comparisons': run_dir / 'comparisons',
        'samples': run_dir / 'samples',
        'logs': run_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs 