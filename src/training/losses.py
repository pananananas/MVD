from torchvision.transforms import Normalize
from torchvision.models import VGG16_Weights
import torchvision.models as models
import torch.nn.functional as F
from icecream import ic
import torch

class PerceptualLoss:
    def __init__(self, device='cuda'):
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:29].to(device).eval()
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.device = device
        
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def __call__(self, x, y):
        current_device = x.device
        if current_device != self.device:
            self.to(current_device)
            
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        x = self.normalize(x)
        y = self.normalize(y)
        
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
        return F.mse_loss(x_features, y_features)
    
    def to(self, device):
        self.device = device
        self.vgg = self.vgg.to(device)
        return self


def compute_losses(
    noise_pred, 
    noise, 
    noisy_latents=None,
    timesteps=None,
    target_latents=None, 
    vae=None, 
    scheduler=None,
    perceptual_loss_fn=None, 
    ssim_loss_fn=None, 
    config=None
):
    
    noise_loss = F.mse_loss(noise_pred, noise)
    total_loss = noise_loss
    
    device = noise_loss.device
    zero_tensor = torch.tensor(0.0, device=device)
    metrics = {
        'latent_recon_loss': zero_tensor,
        'pixel_recon_loss': zero_tensor,
        'perceptual_loss': zero_tensor,
        'ssim_loss': zero_tensor,
        'ssim_value': zero_tensor
    }
    
    if noisy_latents is not None and timesteps is not None and target_latents is not None and vae is not None and scheduler is not None:
        with torch.no_grad():
            try:
                alpha_t = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                beta_t = 1.0 - alpha_t
                
                denoised_latents = (noisy_latents - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
                
                metrics['latent_recon_loss'] = F.mse_loss(denoised_latents.float(), target_latents.float()).detach()
                
                denoised_images = vae.decode(denoised_latents / vae.config.scaling_factor).sample
                target_images = vae.decode(target_latents / vae.config.scaling_factor).sample
                
                metrics['pixel_recon_loss'] = F.mse_loss(denoised_images.float(), target_images.float()).detach()
                
                if perceptual_loss_fn is not None:
                    metrics['perceptual_loss'] = perceptual_loss_fn(denoised_images.float(), target_images.float()).detach()
                
                if ssim_loss_fn is not None:
                    ssim_val = ssim_loss_fn(denoised_images.float(), target_images.float()).detach()
                    metrics['ssim_value'] = ssim_val
                    metrics['ssim_loss'] = 1.0 - ssim_val
            
            except Exception as e:
                ic(f"Exception during auxiliary metric computation: {e}") 
    
    return {
        'total_loss': total_loss,
        'noise_loss': noise_loss,
        **metrics
    } 
