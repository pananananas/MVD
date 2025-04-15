from .losses import PerceptualLoss, compute_losses
from pytorch_lightning import LightningModule
from pytorch_msssim import SSIM
from PIL import Image
import logging
import wandb
import torch

logger = logging.getLogger(__name__)

class MVDLightningModule(LightningModule):
    def __init__(
        self,
        pipeline,
        config,
        dirs
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
        
        # freeze base UNet
        for name, param in self.unet.named_parameters():
            if not any(x in name for x in ['camera_encoder', 'down_modulators', 'up_modulators', 'mid_modulator']):
                param.requires_grad = False
        
        # freeze VAE
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        
        self.dirs = dirs
        self.comparison_dir = self.dirs['comparisons']
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
        
        self.ssim = SSIM(data_range=2.0, size_average=True)
        self.perceptual_loss = PerceptualLoss(device="cpu")
        
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
            source_image_latents=source_latents,
        ).sample
        
        alpha_t = self.scheduler.alphas_cumprod[timesteps]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        denoised_latents = (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        # clamping to prevent extreme values in VAE
        with torch.no_grad():
            if denoised_latents.abs().max().item() > 10.0:
                logger.warning(f"Extreme denoised latent values: {denoised_latents.abs().max().item():.4f}")
                # denoised_latents = torch.clamp(denoised_latents, -10.0, 10.0)
        
            if denoised_latents.abs().max().item() > 10:
                logger.warning(f"WARNING: Denoised latents have extreme values: max abs = {denoised_latents.abs().max().item():.4f}")
        
        return noise_pred, noise, denoised_latents, target_latents, source_latents
    

    def training_step(self, batch, batch_idx):
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
        
        self.log("train/noise_loss", losses['noise_loss'], on_step=True, prog_bar=True)
        
        if batch_idx % self.config.get('metrics_log_interval', 100) == 0:
            for name, value in losses.items():
                if name not in ['total_loss', 'noise_loss', 'decoded_images']:
                    if torch.is_tensor(value):
                        self.log(f"train_metrics/{name}", value.item(), on_step=True)
                    else:
                        self.log(f"train_metrics/{name}", value, on_step=True)
        
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
            if name != 'decoded_images':
                if torch.is_tensor(value):
                    self.log(f"val/{name}", value.item(), on_step=False, on_epoch=True)
                else:
                    self.log(f"val/{name}", value, on_step=False, on_epoch=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items() if k != 'decoded_images'}
        self.validation_step_outputs.append(step_output)
        
        self._save_generated_samples(batch, batch_idx, self.current_epoch)
        
        return losses['total_loss']
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.unet.parameters()),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        warmup_steps = int(0.1 * self.trainer.estimated_stepping_batches)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=warmup_steps / self.trainer.estimated_stepping_batches,
            div_factor=25,
            final_div_factor=10000
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    

    def _save_generated_samples(self, batch, batch_idx, epoch):
        print(f"Saving generated samples for epoch {epoch} in batch {batch_idx}")
        
        with torch.no_grad():
            source_camera = batch.get('source_camera', None)
            target_camera = batch.get('target_camera', None)
            source_images = batch.get('source_image', None)
            
            # Add detailed logging
            logger.info(f"Sample generation - prompt: {batch['prompt']}")
            if source_camera is not None:
                logger.info(f"Source camera shape: {source_camera.shape}")
            if target_camera is not None:
                logger.info(f"Target camera shape: {target_camera.shape}")
            if source_images is not None:
                logger.info(f"Source images shape: {source_images.shape}")
            
            if target_camera is not None and hasattr(target_camera, 'to'):
                target_camera = target_camera.to(self.device)
            
            if source_camera is not None and hasattr(source_camera, 'to'):
                source_camera = source_camera.to(self.device)
                
            if source_images is not None and hasattr(source_images, 'to'):
                source_images = source_images.to(self.device)
            
            try:
                print(f"Generating images for batch {batch_idx} in epoch {epoch}")
                images = self.pipeline(
                    prompt=batch['prompt'],
                    num_inference_steps=20,
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_images=source_images,
                    num_images_per_prompt=1,
                    guidance_scale = 1.0,
                    ref_scale = 0.1,
                    output_type="np"
                )["images"]
                print(f"Generation was successful for batch {batch_idx} in epoch {epoch}, saving samples")
                
                save_dir = self.dirs['samples'] / f"epoch_{epoch}"
                save_dir.mkdir(exist_ok=True)
                
                logger.info(f"Processing batch {batch_idx} in epoch {epoch}")
                
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
                
                    
                    if batch_idx % 500 == 0:
                        logger.info(f"Logging WandB images for batch {batch_idx}, sample {i}")
                        wandb.log({
                            f"samples/epoch_{epoch}_batch_{batch_idx}": [
                                wandb.Image(source_img, caption=f"Source {i}"),
                                wandb.Image(target_img, caption=f"Target {i}"),
                                wandb.Image(generated_img, caption=f"Generated {i}")
                            ]
                        })
                print(f'Saving samples was successful') 

            except Exception as e:
                logger.error(f"Error in sample generation: {str(e)}")
                print(f"Prompt: {batch['prompt']}")
                if source_camera is not None:
                    print(f"Source camera type: {type(source_camera)}, shape: {source_camera.shape}")
                if target_camera is not None:
                    print(f"Target camera type: {type(target_camera)}, shape: {target_camera.shape}")
                if source_images is not None:
                    print(f"Source images type: {type(source_images)}, shape: {source_images.shape}") 