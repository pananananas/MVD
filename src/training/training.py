from .losses import PerceptualLoss, compute_losses
from pytorch_lightning import LightningModule
from src.utils import create_output_dirs
from pytorch_msssim import SSIM
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import wandb
import torch

logger = logging.getLogger(__name__)

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
        
        total_params = sum(p.numel() for p in self.unet.parameters())
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        percentage = trainable_params / total_params * 100
        
        logger.info(f"Total UNet parameters: {total_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Training: {percentage:.2f}% of the model")
        
        self.dirs = create_output_dirs(output_dir)
        self.comparison_dir = Path(output_dir) / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize both loss functions properly
        self.ssim = SSIM(data_range=2.0, size_average=True)  # range [-1,1]
        self.perceptual_loss = PerceptualLoss(device="cpu")  # Will move to correct device later
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        

    def forward(self, batch):
        source_images = batch['source_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)

        source_latents = self.vae.encode(source_images).latent_dist.sample()
        source_latents = source_latents * self.vae.config.scaling_factor
        
        target_latents = self.vae.encode(target_images).latent_dist.sample()
        target_latents = target_latents * self.vae.config.scaling_factor
        
        prompts = batch['prompt']
        
        source_camera = batch.get('source_camera', None)
        target_camera = batch.get('target_camera', None)
        
        if source_camera is not None and hasattr(source_camera, 'to'):
            source_camera = source_camera.to(self.device)
        if target_camera is not None and hasattr(target_camera, 'to'):
            target_camera = target_camera.to(self.device)
            
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        noise = torch.randn_like(source_latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (source_latents.shape[0],),
            device=self.device
        )
        noisy_latents = self.scheduler.add_noise(source_latents, noise, timesteps)
        
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            source_camera=source_camera,
            target_camera=target_camera,
            source_image_latents=source_latents,  # Pass source latents for conditioning
        ).sample
        
        alpha_t = self.scheduler.alphas_cumprod[timesteps]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        denoised_latents = (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        return noise_pred, noise, denoised_latents, target_latents, source_latents
    

    def training_step(self, batch, batch_idx):
        noise_pred, noise, denoised_latents, target_latents, source_latents = self.forward(batch)

        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            denoised_latents=denoised_latents,
            target_latents=target_latents,
            vae=self.vae,  # Pass the VAE for decoding
            perceptual_loss_fn=self.perceptual_loss,
            ssim_loss_fn=self.ssim,
            config=self.config
        )
        
        for name, value in losses.items():
            if name != 'decoded_images' and name != 'total_loss':  # Skip non-scalar items
                self.log(f"train/{name}", value.item() if torch.is_tensor(value) else value, 
                       on_step=True, prog_bar=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items() if k != 'decoded_images'}
        self.training_step_outputs.append(step_output)
        
        return losses['total_loss']
    

    def validation_step(self, batch, batch_idx):
        noise_pred, noise, denoised_latents, target_latents, source_latents = self.forward(batch)
        
        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            denoised_latents=denoised_latents,
            target_latents=target_latents,
            vae=self.vae,
            perceptual_loss_fn=self.perceptual_loss,
            ssim_loss_fn=self.ssim,
            config=self.config
        )
        
        for name, value in losses.items():
            if name != 'decoded_images' and name != 'total_loss':  # Skip non-scalar items
                self.log(f"val/{name}", value.item() if torch.is_tensor(value) else value, 
                       on_step=False, on_epoch=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items() if k != 'decoded_images'}
        self.validation_step_outputs.append(step_output)
        
        if batch_idx % self.config.get('sample_interval', 100) == 0:
            self._save_generated_samples(batch, batch_idx, self.current_epoch)
            # self._save_latent_comparisons(losses['decoded_images'], batch_idx, self.current_epoch)
        
        return losses['total_loss']
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.unet.parameters()),
            lr=self.config['learning_rate']
        )
        
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
    
# Visualization functions for debugging:

    def _save_generated_samples(self, batch, batch_idx, epoch):
        """
        Generate and save samples during training.
        
        Parameters:
            batch: The current batch
            batch_idx: The batch index
            epoch: The current epoch
        """
        with torch.no_grad():
            source_camera = batch.get('source_camera', None)
            target_camera = batch.get('target_camera', None)
            source_images = batch.get('source_image', None)
            
            if target_camera is not None and hasattr(target_camera, 'to'):
                target_camera = target_camera.to(self.device)
            
            if source_camera is not None and hasattr(source_camera, 'to'):
                source_camera = source_camera.to(self.device)
                
            if source_images is not None and hasattr(source_images, 'to'):
                source_images = source_images.to(self.device)
            
            try:
                images = self.pipeline(
                    prompt=batch['prompt'],
                    num_inference_steps=20,
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_images=source_images,  # Pass source images for conditioning
                    num_images_per_prompt=1,
                    output_type="np"
                )["images"]
                
                save_dir = self.dirs['samples'] / f"epoch_{epoch}"
                save_dir.mkdir(exist_ok=True)
                
                for i, (source, target, generated) in enumerate(zip(
                    batch['source_image'],
                    batch['target_image'],
                    images
                )):
                    source_np = ((source.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                    target_np = ((target.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                    
                    if isinstance(generated, torch.Tensor):
                        generated_np = (generated.cpu().permute(1, 2, 0) * 255).numpy().astype('uint8')
                    else:
                        generated_np = (generated * 255).astype('uint8')
                    
                    source_img    = Image.fromarray(source_np)
                    target_img    = Image.fromarray(target_np)
                    generated_img = Image.fromarray(generated_np)
                    
                    source_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_source.png')
                    target_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_target.png')
                    generated_img.save(save_dir / f'batch_{batch_idx}_sample_{i}_generated.png')
                
                wandb.log({
                    f"samples/epoch_{epoch}_batch_{batch_idx}": [
                        wandb.Image(img) for img in images[:4]
                    ]
                })
            except Exception as e:
                logger.error(f"Error in sample generation: {str(e)}")
                print(f"Prompt: {batch['prompt']}")
                if source_camera is not None:
                    print(f"Source camera type: {type(source_camera)}")
                if target_camera is not None:
                    print(f"Target camera type: {type(target_camera)}")
                if source_images is not None:
                    print(f"Source images shape: {source_images.shape}") 


    def save_latent_comparisons(self, decoded_images, batch_idx, epoch, prefix="train"):
        """
        Save comparison between decoded latents for visual inspection
        
        Args:
            decoded_images: Dictionary with 'denoised' and 'target' tensors
            batch_idx: Current batch index
            epoch: Current epoch
            prefix: Prefix for saved files ('train' or 'val')
        """
        with torch.no_grad():
            save_dir = self.comparison_dir / f"epoch_{epoch}"
            save_dir.mkdir(exist_ok=True, parents=True)
            
            denoised_images = decoded_images['denoised']
            target_images = decoded_images['target']
            
            max_images = min(4, denoised_images.shape[0])
            
            for i in range(max_images):
                denoised_np = ((denoised_images[i].cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                target_np = ((target_images[i].cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
                
                h, w, c = denoised_np.shape
                comparison = np.zeros((h, w*2, c), dtype=np.uint8)
                comparison[:, :w, :] = denoised_np
                comparison[:, w:, :] = target_np
                
                comparison_img = Image.fromarray(comparison)
                comparison_img.save(save_dir / f'{prefix}_batch_{batch_idx}_sample_{i}_comparison.png')
                
                Image.fromarray(denoised_np).save(save_dir / f'{prefix}_batch_{batch_idx}_sample_{i}_denoised.png')
                Image.fromarray(target_np).save(save_dir / f'{prefix}_batch_{batch_idx}_sample_{i}_target.png')
                
            wandb.log({
                f"{prefix}_comparisons/epoch_{epoch}_batch_{batch_idx}": [
                    wandb.Image(
                        np.hstack([
                            ((denoised_images[i].cpu().permute(1, 2, 0) + 1) / 2).numpy(),
                            ((target_images[i].cpu().permute(1, 2, 0) + 1) / 2).numpy()
                        ])
                    ) for i in range(max_images)
                ]
            }) 