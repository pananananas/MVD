# This needs adjusting to the new dataset and architecture

from .losses import PerceptualLoss, compute_losses
from pytorch_lightning import LightningModule
from src.utils import create_output_dirs
from pytorch_msssim import SSIM
from PIL import Image
import wandb
import torch

class MVDLightningModule(LightningModule):
    def __init__(
        self,
        pipeline,
        config,
        output_dir="outputs"
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['pipeline'])
        
        # Register pipeline components as modules
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.pipeline = pipeline
        
        # Freeze base UNet parameters
        for name, param in self.unet.named_parameters():
            # Only train the added multi-view components
            if not any(x in name for x in ['camera_encoder', 'down_modulators', 'up_modulators', 'mid_modulator']):
                param.requires_grad = False
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.unet.parameters())
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"Total UNet parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Training: {trainable_params / total_params * 100:.2f}% of the model")
        
        # Create output directories
        self.dirs = create_output_dirs(output_dir)
        
        # Initialize SSIM loss
        self.ssim = SSIM(data_range=2.0, size_average=True)  # range [-1,1]
        
        # Perceptual loss initialized in setup
        self.perceptual_loss = None
        
        # For logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def setup(self, stage=None):
        # Initialize perceptual loss
        if self.perceptual_loss is None:
            self.perceptual_loss = PerceptualLoss(device=self.device)
        self.pipeline.to(self.device)
        torch.set_float32_matmul_precision('high')
    
    def on_train_epoch_start(self):
        self.training_step_outputs = []
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
    
    def on_train_epoch_end(self):
        # Calculate epoch-level metrics
        epoch_losses = {
            'total_loss': 0.0,
            'noise_loss': 0.0,
            'recon_loss': 0.0,
            'perceptual_loss': 0.0,
            'ssim_loss': 0.0,
            'geometric_loss': 0.0,
            'ssim_value': 0.0,
        }
        
        for output in self.training_step_outputs:
            for key in epoch_losses:
                epoch_losses[key] += output[key]
        
        # Average the losses
        num_steps = len(self.training_step_outputs)
        if num_steps > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_steps
                self.log(f"train/{key}_epoch", epoch_losses[key], on_epoch=True, sync_dist=True)
        
        self.training_step_outputs = []
    
    def on_validation_epoch_end(self):
        # Calculate epoch-level metrics
        epoch_losses = {
            'total_loss': 0.0,
            'noise_loss': 0.0,
            'recon_loss': 0.0,
            'perceptual_loss': 0.0,
            'ssim_loss': 0.0,
            'geometric_loss': 0.0,
            'ssim_value': 0.0,
        }
        
        for output in self.validation_step_outputs:
            for key in epoch_losses:
                epoch_losses[key] += output[key]
        
        # Average the losses
        num_steps = len(self.validation_step_outputs)
        if num_steps > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_steps
                self.log(f"val/{key}", epoch_losses[key], on_epoch=True, sync_dist=True)
        
        self.validation_step_outputs = []
    
    def forward(self, batch):
        # Move batch to device
        source_images = batch['source_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)
        source_camera = {k: v.to(self.device) for k, v in batch['source_camera'].items()}
        target_camera = {k: v.to(self.device) for k, v in batch['target_camera'].items()}
        
        # Prepare prompt
        batch_size = source_images.shape[0]
        prompts = [self.config['prompt']] * batch_size
        
        # Tokenize with proper batch size
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode prompt
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        # Forward pass
        noise = torch.randn_like(source_images)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (source_images.shape[0],),
            device=self.device
        )
        noisy_images = self.scheduler.add_noise(source_images, noise, timesteps)
        
        # Add extra channel for noise prediction (alpha channel)
        noisy_images = torch.cat([noisy_images, torch.zeros_like(noisy_images[:, :1])], dim=1)
        
        # Predict noise
        noise_pred = self.unet(
            sample=noisy_images,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            source_camera=source_camera,
            target_camera=target_camera,
        ).sample
        
        # Denoise the image
        alpha_t = self.scheduler.alphas_cumprod[timesteps]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        denoised_images = (noisy_images - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        return noise_pred, noise, denoised_images, target_images, source_images
    
    def training_step(self, batch, batch_idx):
        noise_pred, noise, denoised_images, target_images, source_images = self.forward(batch)
        
        # Compute losses
        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            denoised_images=denoised_images,
            target_images=target_images,
            source_images=source_images,
            perceptual_loss_fn=self.perceptual_loss,
            ssim_loss_fn=self.ssim,
            config=self.config
        )
        
        # Log step metrics
        for name, value in losses.items():
            self.log(f"train/{name}_step", value.item() if torch.is_tensor(value) else value, 
                    on_step=True, prog_bar=True)
        
        # Save for epoch-level computation
        self.training_step_outputs.append(
            {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        )
        
        # Save samples periodically
        if batch_idx % self.config.get('sample_interval', 100) == 0:
            self._save_generated_samples(batch, batch_idx, self.current_epoch)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        noise_pred, noise, denoised_images, target_images, source_images = self.forward(batch)
        
        # Compute losses
        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            denoised_images=denoised_images,
            target_images=target_images,
            source_images=source_images,
            perceptual_loss_fn=self.perceptual_loss,
            ssim_loss_fn=self.ssim,
            config=self.config
        )
        
        # Save for epoch-level computation
        self.validation_step_outputs.append(
            {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        )
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        # Only optimize UNet parameters
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.unet.parameters()),
            lr=self.config['learning_rate']
        )
        
        # Configure learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.trainer.estimated_stepping_batches,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def _save_generated_samples(self, batch, batch_idx, epoch):
        with torch.no_grad():
            # Generate images
            batch_size = len(batch['source_image'])
            images = self.pipeline(
                prompt=[self.config['prompt']] * batch_size,
                num_inference_steps=20,
                source_camera=batch['source_camera'],
                target_camera=batch['target_camera'],
                num_images_per_prompt=1,
                output_type="np"
            ).images
            
            # Save images
            save_dir = self.dirs['samples'] / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True)
            
            for i, (source, target, generated) in enumerate(zip(
                batch['source_image'],
                batch['target_image'],
                images
            )):
                # Convert tensors to numpy arrays (properly denormalized)
                source_np = ((source.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                target_np = ((target.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                generated_np = (generated * 255).astype('uint8')
                
                source_img    = Image.fromarray(source_np)
                target_img    = Image.fromarray(target_np)
                generated_img = Image.fromarray(generated_np)
                source_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_source.png')
                target_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_target.png')
                generated_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_generated.png')
            
            # Log images to wandb
            wandb.log({
                f"samples/epoch_{epoch}_batch_{batch_idx}": [
                    wandb.Image(img) for img in images[:4]  # Log first 4 images
                ]
            }) 