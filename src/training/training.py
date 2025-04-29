from .losses import PerceptualLoss, compute_losses
from pytorch_lightning import LightningModule
from pytorch_msssim import SSIM
from icecream import ic
from PIL import Image
import torch.nn.functional as F
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
        
        self.grad_norms = {}
        self.param_norms = {}
        self.last_batch = None

        self.metrics_log_interval = self.config.get('metrics_log_interval', self.config.get('sample_interval', 10))
        
        self.monitored_param_groups = ['camera_encoder', 'down_modulators', 'up_modulators', 'mid_modulator']
        
        # store parameters by group for gradient monitoring
        self.param_groups = {group: [] for group in self.monitored_param_groups}
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                for group in self.monitored_param_groups:
                    if group in name:
                        self.param_groups[group].append((name, param))
                        break

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
        
        return noise_pred, noise, noisy_latents, timesteps, target_latents, source_latents
    

    def training_step(self, batch, batch_idx):
        self.last_batch = batch
        
        noise_pred, noise, noisy_latents, timesteps, target_latents, source_latents = self.forward(batch)

        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            target_latents=target_latents,
            vae=self.vae,
            scheduler=self.scheduler,
            perceptual_loss_fn=self.perceptual_loss,
            ssim_loss_fn=self.ssim,
            config=self.config
        )
        
        self.log("train/noise_loss", losses['noise_loss'], on_step=True, prog_bar=True)
        
        if batch_idx % self.metrics_log_interval == 0:
            for name, value in losses.items():
                if name not in ['total_loss', 'noise_loss', 'decoded_images']:
                    if torch.is_tensor(value):
                        self.log(f"train_metrics/{name}", value.item(), on_step=True)
                    else:
                        self.log(f"train_metrics/{name}", value, on_step=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items() if k != 'decoded_images'}
        self.training_step_outputs.append(step_output)
        
        return losses['noise_loss']
    

    def validation_step(self, batch, batch_idx):
        noise_pred, noise, noisy_latents, timesteps, target_latents, source_latents = self.forward(batch)
        
        losses = compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            target_latents=target_latents,
            vae=self.vae,
            scheduler=self.scheduler,
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
        
        generated_images_tensor = None
        try:
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
                
                pipeline_output = self.pipeline(
                    prompt=batch['prompt'],
                    num_inference_steps=20,
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_images=source_images,
                    num_images_per_prompt=1,
                    guidance_scale = 1.0,
                    ref_scale = 0.1,
                    output_type="pt"
                )
                generated_images_tensor = pipeline_output["images"]
                generated_images_tensor = (generated_images_tensor * 2.0 - 1.0).to(self.device)

        except Exception as e:
            logger.error(f"Error during validation pipeline generation: {e}")
            return losses['noise_loss']

        if generated_images_tensor is not None:
            target_images_tensor = batch['target_image'].to(self.device)
            
            full_metrics = self._calculate_full_image_metrics(generated_images_tensor, target_images_tensor)

            for name, value in full_metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
            step_output.update(full_metrics)

        if generated_images_tensor is not None:
            save_dir = self.dirs['samples'] / f"epoch_{self.current_epoch}"
            save_dir.mkdir(exist_ok=True, parents=True)

            generated_np = ((generated_images_tensor.cpu().permute(0, 2, 3, 1) + 1) / 2 * 255).numpy().astype('uint8')
            source_np = ((batch['source_image'].cpu().permute(0, 2, 3, 1) + 1) / 2 * 255).numpy().astype('uint8')
            target_np = ((batch['target_image'].cpu().permute(0, 2, 3, 1) + 1) / 2 * 255).numpy().astype('uint8')

            for i in range(len(generated_np)):
                base_filename = f'batch_{batch_idx}_sample_{i}'
                gen_pil, src_pil, tgt_pil = self._save_image_set(
                    generated_np[i], source_np[i], target_np[i], save_dir, base_filename
                )

                if batch_idx % 500 == 0:
                    wandb_key = f"val_samples/epoch_{self.current_epoch}_batch_{batch_idx}_sample_{i}"
                    self._log_wandb_comparison(gen_pil, src_pil, tgt_pil, wandb_key)

        return losses['noise_loss']
    

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
    

    def _calculate_full_image_metrics(self, generated_images_tensor, target_images_tensor):
        metrics = {}
        try:
            with torch.no_grad():
                gen_img = generated_images_tensor.float()
                tgt_img = target_images_tensor.float()

                metrics['full_pixel_recon_loss'] = F.mse_loss(gen_img, tgt_img).item()

                if self.perceptual_loss:
                    metrics['full_perceptual_loss'] = self.perceptual_loss(gen_img, tgt_img).item()

                if self.ssim:
                    ssim_val = self.ssim(gen_img, tgt_img).item()
                    metrics['full_ssim_value'] = ssim_val
                    metrics['full_ssim_loss'] = 1.0 - ssim_val

        except Exception as e:
            logger.error(f"Error calculating full image metrics: {e}")
        return metrics


    def _save_image_set(self, generated_img_np, source_img_np, target_img_np, save_dir, base_filename):
        try:
            source_img = Image.fromarray(source_img_np)
            target_img = Image.fromarray(target_img_np)
            generated_img = Image.fromarray(generated_img_np)

            source_img.save(save_dir / f'{base_filename}_source.png')
            target_img.save(save_dir / f'{base_filename}_target.png')
            generated_img.save(save_dir / f'{base_filename}_generated.png')
            return generated_img, source_img, target_img
        except Exception as e:
            logger.error(f"Error saving image set {base_filename}: {e}")
            return None, None, None

    def _log_wandb_comparison(self, generated_img_pil, source_img_pil, target_img_pil, wandb_key):
        if generated_img_pil and source_img_pil and target_img_pil and self.logger and hasattr(self.logger.experiment, 'log'):
            try:
                wandb.log({
                    wandb_key: [
                        wandb.Image(source_img_pil, caption="Source"),
                        wandb.Image(target_img_pil, caption="Target"),
                        wandb.Image(generated_img_pil, caption="Generated")
                    ]
                }, step=self.global_step)
            except Exception as e:
                logger.error(f"Error logging WandB comparison {wandb_key}: {e}")
    
    
    
    def on_after_backward(self):
        if self.global_step % self.metrics_log_interval == 0:
            for group_name, params in self.param_groups.items():
                group_grad_norm = 0.0
                group_param_norm = 0.0
                group_grad_max = 0.0
                group_grad_min = float('inf')
                grad_histogram_values = []
                
                for name, param in params:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        param_norm = param.norm().item()
                        
                        group_grad_norm += grad_norm
                        group_param_norm += param_norm
                        
                        grad_max = param.grad.max().item()
                        grad_min = param.grad.min().item()
                        
                        if grad_max > group_grad_max:
                            group_grad_max = grad_max
                        if grad_min < group_grad_min:
                            group_grad_min = grad_min
                        
                        if self.global_step % (self.metrics_log_interval * 5) == 0:
                            self.log(f"gradients/{group_name}/{name}_norm", grad_norm, on_step=True)
                            self.log(f"parameters/{group_name}/{name}_norm", param_norm, on_step=True)
                        
                        if self.global_step % (self.metrics_log_interval * 10) == 0:
                            flat_grads = param.grad.flatten().cpu().numpy()
                            if len(flat_grads) > 1000:
                                indices = torch.randperm(len(flat_grads))[:1000]
                                flat_grads = flat_grads[indices]
                            grad_histogram_values.extend(flat_grads)
                
                self.log(f"gradients/{group_name}/total_norm", group_grad_norm, on_step=True)
                self.log(f"parameters/{group_name}/total_norm", group_param_norm, on_step=True)
                self.log(f"gradients/{group_name}/max_value", group_grad_max, on_step=True)
                self.log(f"gradients/{group_name}/min_value", group_grad_min, on_step=True)
                
                if self.global_step % (self.metrics_log_interval * 10) == 0 and grad_histogram_values and self.logger:
                    if hasattr(self.logger, "experiment"):
                        self.logger.experiment.log({
                            f"gradients/{group_name}/histogram": wandb.Histogram(
                                np.array(grad_histogram_values)
                            ),
                            "global_step": self.global_step
                        })
            
            for group_name in self.monitored_param_groups:
                self.grad_norms[group_name] = group_grad_norm
                self.param_norms[group_name] = group_param_norm
            
            for group_name in self.monitored_param_groups:
                if group_name in self.grad_norms and group_name in self.param_norms:
                    grad_to_param_ratio = self.grad_norms[group_name] / (self.param_norms[group_name] + 1e-8)
                    self.log(f"gradients/{group_name}/grad_to_param_ratio", grad_to_param_ratio, on_step=True)
            
            if hasattr(self.trainer, "optimizers"):
                optimizer = self.trainer.optimizers[0]
                for i, param_group in enumerate(optimizer.param_groups):
                    if "lr" in param_group:
                        self.log(f"optimizer/param_group_{i}_lr", param_group["lr"], on_step=True)

    def on_train_epoch_end(self):
        avg_metrics = {}
        for metric in self.training_step_outputs[0].keys():
            avg_metrics[metric] = sum(output[metric] for output in self.training_step_outputs) / len(self.training_step_outputs)
            self.log(f"train_epoch/{metric}", avg_metrics[metric])
        
        self.training_step_outputs.clear() 