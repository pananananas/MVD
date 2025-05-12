from src.models.attention import ImageCrossAttentionProcessor
from .losses import PerceptualLoss, compute_losses
from pytorch_lightning import LightningModule
from pytorch_msssim import SSIM
import torch.nn.functional as F
from typing import Optional
from icecream import ic
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
        dirs,
        debug_log_file_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['pipeline'])
        
        self.debug_log_file_path = debug_log_file_path
        
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.pipeline = pipeline
        
        # 1. Freeze VAE and Text Encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

        # 2. Freeze all parameters of self.unet
        for param in self.unet.parameters():
            param.requires_grad = True

        # 3. Unfreeze ImageCrossAttentionProcessor
        if self.unet.use_image_conditioning:
            
            found_attn_procs = False
            for name, module in self.unet.base_unet.named_modules():
                if hasattr(module, 'processor') and isinstance(module.processor, ImageCrossAttentionProcessor):
                    found_attn_procs = True
                    for sub_name, param in module.processor.named_parameters():
                        param.requires_grad = True
            if found_attn_procs:
                ic(f"Finished unfreezing ImageCrossAttentionProcessor parameters. Amount of trainable parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad)}")
            else:
                ic("WARNING - No ImageCrossAttentionProcessor instances found in self.unet.base_unet, though image conditioning is ON.")
        else:
            ic("Image conditioning is OFF. ImageCrossAttentionProcessors parameters remain frozen.")

        # 4. Unfreeze camera_encoder parameters if it exists and camera embeddings are enabled in MultiViewUNet
        if self.unet.camera_encoder is not None and self.unet.use_camera_conditioning:
            ic("Camera embeddings are ON. Attempting to unfreeze self.unet.camera_encoder parameters.")
            for name, param in self.unet.camera_encoder.named_parameters():
                param.requires_grad = True
                # ic(f"  Unfroze camera_encoder.{name}") # Verbose
            ic(f"Finished unfreezing camera_encoder parameters. Amount of trainable parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad)}")
        
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
        self.modulation_log_interval = self.config.get('modulation_log_interval', self.metrics_log_interval * 5)
        
        self.monitored_param_groups = ['camera_encoder', 'image_attention_processor', 'down_modulators', 'up_modulators', 'mid_modulator']
        
        # store parameters by group for gradient monitoring
        self.param_groups = {group: [] for group in self.monitored_param_groups}
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                # ic(f"MVDLightningModule.__init__: Found trainable param for param_groups: {name}") # This was commented out by user, keeping it so.
                if name.startswith('camera_encoder.'): # Check if the parameter name indicates it's from the camera_encoder
                     self.param_groups['camera_encoder'].append((name, param))
                elif '.processor.' in name: # Heuristic for ImageCrossAttentionProcessor params
                     self.param_groups['image_attention_processor'].append((name, param))
                else: # Fallback for other modulators if they are made trainable and identified by name
                    for group_name_key in ['down_modulators', 'up_modulators', 'mid_modulator']:
                        if group_name_key in name:
                            self.param_groups[group_name_key].append((name, param))
                            break
                        
        ic(f"Populated self.param_groups: {[(group, len(params)) for group, params in self.param_groups.items()]}")

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
        
        noise = torch.randn_like(target_latents)
        
        timesteps = torch.randint(
            0, 
            self.scheduler.config.num_train_timesteps, 
            (target_latents.shape[0],), 
            device=self.device
        )
        
        noisy_latents = self.scheduler.add_noise(target_latents, noise, timesteps)
        
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
                if name not in ['total_loss', 'noise_loss']:
                    log_name = f"train_metrics/{name}"
                    if torch.is_tensor(value):
                        self.log(log_name, value.item(), on_step=True)
                    else:
                        self.log(log_name, value, on_step=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items()}
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
            if name != 'total_loss':
                log_name = f"val/{name}"
                if torch.is_tensor(value):
                    self.log(log_name, value.item(), on_step=False, on_epoch=True)
                else:
                    self.log(log_name, value, on_step=False, on_epoch=True)
        
        step_output = {k: v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items()}
        self.validation_step_outputs.append(step_output)
        
        generated_images_tensor = None

        try:
            self.unet.eval()
            self.vae.eval()
            self.text_encoder.eval()
            if hasattr(self.unet, 'image_encoder') and self.unet.image_encoder is not None:
                self.unet.image_encoder.eval()
            if hasattr(self.unet, 'camera_encoder') and self.unet.camera_encoder is not None:
                self.unet.camera_encoder.eval()

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
                    output_type="pt",
                    use_camera_embeddings=self.unet.use_camera_conditioning,
                    use_image_conditioning=self.unet.use_image_conditioning,
                    debug_log_file_path=self.debug_log_file_path
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

                if batch_idx % 50 == 0:
                    self._log_wandb_comparison(gen_pil, src_pil, tgt_pil, self.current_epoch, batch_idx )

        return losses['noise_loss']
    

    def configure_optimizers(self):
        ic("MVDLightningModule.configure_optimizers: Checking parameters for optimizer.")
        trainable_params = []
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        if not trainable_params:
            ic("!!! No trainable parameters found for the optimizer !!!")
        else:
            ic(f"Found {len(trainable_params)} trainable parameters for the optimizer.")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        return optimizer
        
        # warmup_steps = int(0.1 * self.trainer.estimated_stepping_batches)
        # ic(f"Total estimated stepping batches for OneCycleLR: {self.trainer.estimated_stepping_batches}, warmup_steps: {warmup_steps}")
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.config['learning_rate'],
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=warmup_steps / self.trainer.estimated_stepping_batches,
        #     div_factor=25,
        #     final_div_factor=10000
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }
    

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

    def _log_wandb_comparison(self, generated_img_pil, source_img_pil, target_img_pil, epoch, batch_idx):
        if generated_img_pil and source_img_pil and target_img_pil and self.logger and hasattr(self.logger.experiment, 'log'):
            try:
                wandb.log({
                    f"samples/epoch_{epoch:03d}_batch_{batch_idx}": [
                        wandb.Image(source_img_pil, caption="Source"),
                        wandb.Image(target_img_pil, caption="Target"),
                        wandb.Image(generated_img_pil, caption="Generated")
                    ]
                })
            except Exception as e:
                logger.error(f"Error logging WandB comparison at global_step {self.global_step}: {e}")
    
    
    
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
                        
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"!!! NaN/Inf gradient detected in {name} !!!")
                            print(f"!!! NaN/Inf gradient detected in {name} !!!")
                        
                        if self.global_step % (self.metrics_log_interval * 10) == 0:
                            self.log(f"gradients/{group_name}/{name}_norm", grad_norm, on_step=True)
                            self.log(f"parameters/{group_name}/{name}_norm", param_norm, on_step=True)
                        
                        if self.global_step % (self.metrics_log_interval * 5) == 0:
                            flat_grads = param.grad.flatten().cpu().numpy()
                            if len(flat_grads) > 1000:
                                indices = np.random.choice(len(flat_grads), 1000, replace=False)
                                flat_grads = flat_grads[indices]
                            grad_histogram_values.extend(flat_grads)
                
                self.log(f"gradients/{group_name}/total_norm", group_grad_norm, on_step=True)
                self.log(f"parameters/{group_name}/total_norm", group_param_norm, on_step=True)
                self.log(f"gradients/{group_name}/max_value", group_grad_max, on_step=True)
                self.log(f"gradients/{group_name}/min_value", group_grad_min, on_step=True)
                
                if self.global_step % (self.metrics_log_interval * 10) == 0 and grad_histogram_values and self.logger:
                    if hasattr(self.logger, "experiment") and hasattr(wandb, "Histogram"):
                        try:
                            self.logger.experiment.log({
                                f"gradients/{group_name}/histogram": wandb.Histogram(
                                    np.array(grad_histogram_values)
                                ),
                                "global_step": self.global_step
                            })
                        except Exception as e:
                            logger.error(f"Failed to log gradient histogram to wandb: {e}")
            
            if 'camera_encoder' in self.param_groups:
                ce_grad_norms = {}
                for name, param in self.param_groups['camera_encoder']:
                    if param.grad is not None:
                        layer_name = name.replace('unet.camera_encoder.', '')
                        if 'rotation_encoder' in layer_name or \
                           'translation_encoder' in layer_name or \
                           'modulators' in layer_name:
                            ce_grad_norms[f"gradients/camera_encoder/{layer_name}_norm"] = param.grad.norm().item()
                if ce_grad_norms:
                    self.log_dict(ce_grad_norms, on_step=True)
            
            for group_name in self.monitored_param_groups:
                self.grad_norms[group_name] = group_grad_norm
                self.param_norms[group_name] = group_param_norm
            
            for group_name in self.monitored_param_groups:
                current_group_grad_norm = sum(p.grad.norm().item() for _, p in self.param_groups[group_name] if p.grad is not None)
                current_group_param_norm = sum(p.norm().item() for _, p in self.param_groups[group_name])
                
                if current_group_param_norm > 1e-8:
                    grad_to_param_ratio = current_group_grad_norm / current_group_param_norm
                    self.log(f"gradients/{group_name}/grad_to_param_ratio", grad_to_param_ratio, on_step=True)
                else:
                    self.log(f"gradients/{group_name}/grad_to_param_ratio", 0.0, on_step=True)
            
            if hasattr(self.trainer, "optimizers"):
                optimizer = self.trainer.optimizers[0]
                for i, param_group in enumerate(optimizer.param_groups):
                    if "lr" in param_group:
                        self.log(f"optimizer/param_group_{i}_lr", param_group["lr"], on_step=True)

        if self.global_step % self.modulation_log_interval == 0:
            if hasattr(self.unet, 'camera_encoder') and hasattr(self.unet.camera_encoder, '_current_modulation_stats'):
                mod_stats = self.unet.camera_encoder._current_modulation_stats
                if mod_stats:
                    log_dict = {}
                    for mod_name, stats in mod_stats.items():
                        for stat_name, value in stats.items():
                            log_dict[f"modulation/{mod_name}/{stat_name}"] = value

                    if log_dict:
                        self.log_dict(log_dict, on_step=True)

                    self.unet.camera_encoder._current_modulation_stats.clear()

    def on_train_epoch_end(self):
        avg_metrics = {}
        for metric in self.training_step_outputs[0].keys():
            avg_metrics[metric] = sum(output[metric] for output in self.training_step_outputs) / len(self.training_step_outputs)
            self.log(f"train_epoch/{metric}", avg_metrics[metric])
        
        self.training_step_outputs.clear() 