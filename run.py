from src.dataset import create_dataloaders
from src.mvd import create_mvd_pipeline
from src.training import MVDTrainer
import wandb
import torch
import os

SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
os.environ['HF_HOME'] = HUGGINGFACE_CACHE

dataset_path = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/co3d/motorcycle"


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = {
        # Model parameters
        "architecture": "runwayml/stable-diffusion-v1-5",
        "torch_dtype": "float32",
        "use_memory_efficient_attention": True,
        "enable_gradient_checkpointing": True,
        
        # Training parameters
        "learning_rate": 0.0002,
        "epochs": 10,
        "batch_size": 3,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
        "val_check_interval": 1,  # Validate every 1 of epoch
        
        # Dataset parameters
        "dataset": "CO3D-motor",
        "image_size": (256, 256),
        "max_angle_diff": 45.0,
        "min_angle_diff": 15.0,
        "max_pairs_per_sequence": 10,
        
        # Loss parameters
        "perceptual_weight": 0.1,
        "ssim_weight": 0.05,
        "geometric_weight": 0.1,
        
        # Checkpoint parameters
        "early_stopping_patience": 20,
        "max_checkpoints": 5,
        "sample_interval": 100,
        
        # Prompt parameters
        "prompt": "a photo of a motorcycle",
        
        # Debug parameters
        "debug": False,
        "debug_num_sequences": 2,
        "debug_max_pairs": 5,
    }
    
    wandb.init(
        project="mvd",
        config=config
    )
    
    train_loader, val_loader = create_dataloaders(
        data_path=dataset_path,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        max_angle_diff=config['max_angle_diff'],
        min_angle_diff=config['min_angle_diff'],
        max_pairs_per_sequence=config['max_pairs_per_sequence'] if not config['debug'] else config['debug_max_pairs'],
        debug_mode=config['debug'],
        debug_num_sequences=config['debug_num_sequences']
    )
    
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=config['architecture'],
        dtype=torch.float32,
        use_memory_efficient_attention=config['use_memory_efficient_attention'],
        enable_gradient_checkpointing=config['enable_gradient_checkpointing'],
        cache_dir=HUGGINGFACE_CACHE,
    )
    pipeline.to(device)
    
    optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=config['learning_rate'])
    
    trainer = MVDTrainer(
        pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        output_dir="outputs"
    )
    
    trainer.train()
    
    wandb.finish()