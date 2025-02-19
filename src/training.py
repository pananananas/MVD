from .utils import EarlyStopping, CheckpointManager, create_output_dirs
from .losses import PerceptualLoss, compute_losses
from pytorch_msssim import SSIM
from tqdm import tqdm
from PIL import Image
import wandb
import torch

class MVDTrainer:
    def __init__(
        self,
        pipeline,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        output_dir="outputs"
    ):
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Create output directories
        self.dirs = create_output_dirs(output_dir)
        
        # Initialize loss functions
        self.perceptual_loss = PerceptualLoss(device=device)
        self.ssim = SSIM(data_range=2.0, size_average=True)  # range [-1,1]
        
        # Initialize training utilities
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 7)
        )
        self.checkpoint_manager = CheckpointManager(
            self.dirs['checkpoints'],
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
        # Load latest checkpoint if exists
        self.start_epoch = 0
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        checkpoint = self.checkpoint_manager.load_latest(self.pipeline, self.optimizer)
        if checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {self.start_epoch}")
    
    def train_step(self, batch, batch_idx):
        # Get losses dictionary
        losses = self.train_step_inner(batch, batch_idx)
        
        # Scale only the total loss for gradient accumulation
        scaled_loss = losses['total_loss'] / self.config['gradient_accumulation_steps']
        scaled_loss.backward()
        
        # Only update weights after accumulating gradients
        if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
            if self.config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.pipeline.unet.parameters(),
                    self.config['max_grad_norm']
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return losses  # Return the original losses dictionary for logging
    
    def train_step_inner(self, batch, batch_idx):
        # Move batch to device
        source_images = batch['source_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)
        source_camera = {k: v.to(self.device) for k, v in batch['source_camera'].items()}
        target_camera = {k: v.to(self.device) for k, v in batch['target_camera'].items()}
        points_3d = [points.to(self.device) for points in batch['points_3d']]
        
        # Prepare prompt
        batch_size = source_images.shape[0]
        prompts = [self.config['prompt']] * batch_size
        
        # Tokenize with proper batch size
        text_input = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode prompt
        text_embeddings = self.pipeline.text_encoder(text_input.input_ids)[0]
        
        # Forward pass
        noise = torch.randn_like(source_images)
        timesteps = torch.randint(
            0,
            self.pipeline.scheduler.config.num_train_timesteps,
            (source_images.shape[0],),
            device=self.device
        )
        noisy_images = self.pipeline.scheduler.add_noise(source_images, noise, timesteps)
        
        # Add extra channel for noise prediction (alpha channel)
        noisy_images = torch.cat([noisy_images, torch.zeros_like(noisy_images[:, :1])], dim=1)
        
        # Predict noise
        noise_pred = self.pipeline.unet(
            sample=noisy_images,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            source_camera=source_camera,
            target_camera=target_camera,
            points_3d=points_3d
        ).sample
        
        # Denoise the image
        alpha_t = self.pipeline.scheduler.alphas_cumprod[timesteps]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        denoised_images = (noisy_images - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        # Compute all losses
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
        
        return losses
    
    @torch.no_grad()
    def validation_step(self, batch):
        self.pipeline.unet.eval()
        losses = self.train_step_inner(batch, 0)  # batch_idx=0 since we don't need it for validation
        self.pipeline.unet.train()
        return losses
    
    def validation_epoch(self):
        val_losses = []
        for batch in self.val_loader:
            losses = self.validation_step(batch)
            val_losses.append(losses['total_loss'].item())
        return sum(val_losses) / len(val_losses)
    
    def save_generated_samples(self, batch, batch_idx, epoch):
        with torch.no_grad():
            # Generate images
            batch_size = len(batch['source_image'])
            images = self.pipeline(
                prompt=[self.config['prompt']] * batch_size,
                num_inference_steps=20,
                source_camera=batch['source_camera'],
                target_camera=batch['target_camera'],
                points_3d=batch['points_3d'],
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
            # wandb.log({
            #     f"samples/epoch_{epoch}_batch_{batch_idx}": [
            #         wandb.Image(img) for img in images[:4]  # Log first 4 images
            #     ]
            # })
    
    def train(self):
        best_val_loss = float('inf')
        global_step = 0
        
        # Initialize learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config['epochs']
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps,
        )
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            self.pipeline.unet.train()
            epoch_losses = []
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                losses = self.train_step(batch, batch_idx)
                epoch_losses.append(losses['total_loss'].item())
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': losses['total_loss'].item(),
                    'avg_loss': sum(epoch_losses) / len(epoch_losses),
                    'lr': scheduler.get_last_lr()[0]
                })
                
                # Log metrics
                wandb.log({
                    f"train/{k}": v for k, v in losses.items()
                })
                wandb.log({
                    'learning_rate': scheduler.get_last_lr()[0],
                    'global_step': global_step
                })
                
                # Save samples periodically
                if batch_idx % self.config.get('sample_interval', 100) == 0:
                    self.save_generated_samples(batch, batch_idx, epoch)
                
                # Update learning rate
                scheduler.step()
            
            # Validation
            val_loss = self.validation_epoch()
            wandb.log({
                'val/loss': val_loss
            })
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint_manager.save(
                    self.pipeline,
                    self.optimizer,
                    epoch,
                    val_loss,
                    metrics={
                        'val_loss': val_loss,
                        'epoch': epoch,
                        'global_step': global_step
                    }
                )
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break 