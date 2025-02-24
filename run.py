from src.dataset import MVDDataModule
from src.mvd import create_mvd_pipeline
from src.training import MVDLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
import os

torch.set_float32_matmul_precision('high')
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
        "architecture": "stabilityai/stable-diffusion-2-1",
        "torch_dtype": "float32",
        "use_memory_efficient_attention": True,
        "enable_gradient_checkpointing": True,
        
        # Training parameters
        "learning_rate": 0.0002,
        "epochs": 10,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
        "val_check_interval": 4,  # Will validate every 4 steps
        
        # Dataset parameters
        "dataset": "CO3D-motor",
        "image_size": (768, 768),
        "max_angle_diff": 45.0,
        "min_angle_diff": 15.0,
        "max_pairs_per_sequence": 10,
        "max_samples_per_epoch": 50,  # Limit samples per epoch
        
        # Loss parameters
        "perceptual_weight": 0.1,
        "ssim_weight": 0.05,
        "geometric_weight": 0.1,
        
        # Checkpoint parameters
        "early_stopping_patience": 20,
        "max_checkpoints": 5,
        "sample_interval": 10,  # Reduced to see more samples within our limited steps
        
        # Prompt parameters
        "prompt": "a photo of a motorcycle",
        
        # Debug parameters
        "debug": False,
        "debug_num_sequences": 2,
        "debug_max_pairs": 5,
    }
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="mvd",
        config=config,
        log_model=True,
    )
    
    # Create data module
    data_module = MVDDataModule(
        data_path=dataset_path,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        max_angle_diff=config['max_angle_diff'],
        min_angle_diff=config['min_angle_diff'],
        max_pairs_per_sequence=config['max_pairs_per_sequence'],
        debug_mode=config['debug'],
        debug_num_sequences=config['debug_num_sequences'],
        debug_max_pairs=config['debug_max_pairs'],
        max_samples_per_epoch=config['max_samples_per_epoch']
    )
    
    # Create pipeline
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=config['architecture'],
        dtype=getattr(torch, config['torch_dtype']),
        use_memory_efficient_attention=config['use_memory_efficient_attention'],
        enable_gradient_checkpointing=config['enable_gradient_checkpointing'],
        cache_dir=HUGGINGFACE_CACHE,
    )
    
    # Create model
    model = MVDLightningModule(
        pipeline=pipeline,
        config=config,
        output_dir="outputs"
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="outputs/checkpoints",
            filename="mvd-{epoch:02d}-{val_loss:.2f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=config['max_checkpoints'],
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=config['early_stopping_patience'],
            mode="min",
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        val_check_interval=config['val_check_interval'],
        log_every_n_steps=1,  # Log every step for better monitoring
        deterministic=False,  # For better performance
        precision="32-true",  # Use true FP32 for stability
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    wandb.finish()