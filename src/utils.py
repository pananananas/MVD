from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import PIL

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


def log_debug(file_path, message):
    """Appends a timestamped message to the debug log file."""
    if file_path:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(file_path, 'a') as f:
                f.write(f"{timestamp} - {message}\n")
        except Exception as e:
            # Avoid crashing the main process due to logging errors
            print(f"[Debug Log Error] Failed to write to {file_path}: {e}")

def load_image(image_path, target_size=(768, 768)):
    image = Image.open(image_path)
    
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert('RGB')
    else:
        image = image.convert('RGB')
    
    image = image.resize(target_size, PIL.Image.Resampling.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)    
    return image_tensor


def create_camera_matrix(position, target, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    
    position = np.array(position)
    target = np.array(target)
    up = np.array(up)
    
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 1e-8:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / forward_norm
    
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    
    new_up = np.cross(right, forward)
    
    rotation = np.zeros((3, 3))
    rotation[:, 0] = right
    rotation[:, 1] = new_up
    rotation[:, 2] = -forward
    
    camera_matrix = np.zeros((3, 4))
    camera_matrix[:3, :3] = rotation
    camera_matrix[:3, 3]  = position
    
    return torch.from_numpy(camera_matrix).float()