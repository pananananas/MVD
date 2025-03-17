from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.datasets.objaverse_dataset import ObjaverseDataModule
from pytorch_lightning.loggers import WandbLogger
from src.training import MVDLightningModule
from src.mvd import create_mvd_pipeline
from pytorch_lightning import Trainer
from pathlib import Path
import argparse
import wandb
import torch
import yaml
import os

torch.set_float32_matmul_precision('high')
# SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
# HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
# os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
# os.environ['HF_HOME'] = HUGGINGFACE_CACHE


dataset_path = "/Users/ewojcik/Code/pwr/MVD/objaverse"

def main(config):

    wandb_logger = WandbLogger(
        project="mvd",
        config=config,
        log_model=True,
    )

    data_module = ObjaverseDataModule(
        data_root=dataset_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=(1024, 1024),
        max_views_per_object=config['max_views_per_object'],
        max_samples=config['max_samples'],
    )
    
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=config['architecture'],
        dtype=getattr(torch, config['torch_dtype']),
        use_memory_efficient_attention=config['use_memory_efficient_attention'],
        enable_gradient_checkpointing=config['enable_gradient_checkpointing'],
        # cache_dir=HUGGINGFACE_CACHE,
    )
    

    model = MVDLightningModule(
        pipeline=pipeline,
        config=config,
        output_dir="outputs"
    )
    
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
    
    trainer = Trainer(
        accelerator="auto",
        devices=config['num_gpus'],
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
    
    trainer.fit(model, data_module)
    
    wandb.finish()


def load_config(config_path):
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    return config


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="MVD Training Script")
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    if isinstance(config['image_size'], list):
        config['image_size'] = tuple(config['image_size'])
    
    if device == "cuda":
        config['num_workers'] = 16
    elif  device == "mps":
        config['num_workers'] = 6
    else:
        config['num_workers'] = 1

    main(config)