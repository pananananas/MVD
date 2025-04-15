from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from src.data.objaverse_dataset import ObjaverseDataModule
from pytorch_lightning.strategies import DDPStrategy
from src.training.training import MVDLightningModule
from src.models.mvd_unet import create_mvd_pipeline
from pytorch_lightning.loggers import WandbLogger
from src.utils import create_output_dirs
from pytorch_lightning import Trainer
from pathlib import Path
import argparse
import wandb
import torch
import yaml
import os


def main(config, cuda, resume_from_checkpoint=None):
    if cuda:
        torch.set_float32_matmul_precision('high')
        SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
        HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
        os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
        os.environ['HF_HOME'] = HUGGINGFACE_CACHE
        dataset_path = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/"
    else:
        dataset_path = "/Users/ewojcik/Code/pwr/MVD/objaverse"

    wandb_logger = WandbLogger(
        project="mvd",
        config=config,
        log_model=True,
    )

    data_module = ObjaverseDataModule(
        data_root=dataset_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=(config['image_size'][0], config['image_size'][1]),
        max_views_per_object=config['max_views_per_object'],
        max_samples=config['max_samples'],
    )
    
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=config['architecture'],
        dtype=getattr(torch, config['torch_dtype']),
        use_memory_efficient_attention=config['use_memory_efficient_attention'],
        enable_gradient_checkpointing=config['enable_gradient_checkpointing'],
        use_camera_embeddings=config.get('use_camera_embeddings', True),
        use_image_conditioning=config.get('use_image_conditioning', True),
        cache_dir=HUGGINGFACE_CACHE if cuda else None,
    )

    dirs = create_output_dirs("outputs")

    model = MVDLightningModule(
        pipeline=pipeline,
        config=config,
        dirs=dirs
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirs['checkpoints'],
        filename="mvd-{epoch:02d}-{val_loss:.2f}",
        monitor="val/noise_loss",
        mode="min",
        save_top_k=config['max_checkpoints'],
        save_last=True,
        every_n_train_steps=1000,
    )
    
    timer_callback = Timer(
        duration={"hours": 47},
        interval="step"
    )
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval='step'),
        timer_callback,
    ]
    
    precision_value = "32" if config['torch_dtype'] == 'float32' else "16"
    print(f"Using PyTorch Lightning precision: {precision_value}")
    
    if config['num_gpus'] > 1:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
    else:
        strategy = "auto"

    trainer = Trainer(
        accelerator="auto",
        devices=config['num_gpus'],
        strategy=strategy,
        max_epochs=config['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        val_check_interval=config['val_check_interval'],
        log_every_n_steps=1,
        deterministic=False,
        precision=precision_value,
    )
    
    trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint or config.get('checkpoint_path'))
    
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

    parser = argparse.ArgumentParser(description="MVD Training Script")
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to configuration file')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    config = load_config(args.config)
    
    if isinstance(config['image_size'], list):
        config['image_size'] = tuple(config['image_size'])
    
    if args.cuda:
        config['num_workers'] = 32
    else:
        config['num_workers'] = 6

    main(config, args.cuda, resume_from_checkpoint=args.resume)