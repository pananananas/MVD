from torchvision.transforms import Normalize
from torchvision.models import VGG16_Weights
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import SSIM
import torch

class PerceptualLoss:
    def __init__(self, device='cuda'):
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:29].to(device).eval()
        # ImageNet normalization
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def __call__(self, x, y):
        # Assuming x, y are in range [-1, 1], convert to [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Normalize for VGG
        x = self.normalize(x)
        y = self.normalize(y)
        
        # Get features
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
        return F.mse_loss(x_features, y_features)


def compute_geometric_consistency(generated, target):
    """
    THIS IS A PLACEHOLDER FOR GEOMETRIC CONSISTENCY LOSS
    TODO: Implement a proper geometric consistency loss
    """
    return F.l1_loss(generated, target)


def compute_losses(noise_pred, noise, denoised_latents, target_latents, vae, perceptual_loss_fn, ssim_loss_fn, config):
    """
    Compute all loss components for MVD training with proper space handling
    
    Args:
        noise_pred: Predicted noise (4-channel latent)
        noise: Ground truth noise (4-channel latent)
        denoised_latents: Generated images after denoising (4-channel latent)
        target_latents: Ground truth target view images (4-channel latent)
        source_latents: Source view images (4-channel latent)
        vae: VAE model to decode latents to pixel space
        perceptual_loss_fn: Instance of PerceptualLoss
        ssim_loss_fn: Instance of SSIM
        config: Training configuration with loss weights
    """

    noise_loss = F.mse_loss(noise_pred, noise)
    latent_recon_loss = F.l1_loss(denoised_latents, target_latents)
    
    decoded_images = {}
    
    with torch.no_grad():
        scaled_denoised = denoised_latents / vae.config.scaling_factor
        scaled_target   = target_latents   / vae.config.scaling_factor
        denoised_images = vae.decode(scaled_denoised).sample
        target_images   = vae.decode(scaled_target).sample
        
        decoded_images['denoised'] = denoised_images
        decoded_images['target']   = target_images
    
    pixel_recon_loss = F.l1_loss(denoised_images, target_images)
    
    ssim_value = torch.tensor(0.0, device=noise_loss.device)
    ssim_loss  = torch.tensor(1.0, device=noise_loss.device)
    perceptual_loss = torch.tensor(0.0, device=noise_loss.device)
    geometric_loss  = compute_geometric_consistency(denoised_images, target_images)
    
    try:
        if ssim_loss_fn is not None:
            ssim_value = ssim_loss_fn(denoised_images, target_images)
            ssim_loss = 1.0 - ssim_value  # Higher SSIM is better, so invert for loss
    except Exception as e:
        print(f"SSIM calculation error: {e}")
    
    try:
        if perceptual_loss_fn is not None:
            perceptual_loss = perceptual_loss_fn(denoised_images, target_images)
    except Exception as e:
        print(f"Perceptual loss calculation error: {e}")
    
    noise_weight        = config.get('noise_weight', 1.0)
    latent_recon_weight = config.get('latent_recon_weight', 0.5)
    pixel_recon_weight  = config.get('pixel_recon_weight', 1.0)
    perceptual_weight   = config.get('perceptual_weight', 0.1)
    ssim_weight         = config.get('ssim_weight', 0.1)
    geometric_weight    = config.get('geometric_weight', 0.5)

    total_loss = (
        noise_weight * noise_loss +
        latent_recon_weight * latent_recon_loss +
        pixel_recon_weight * pixel_recon_loss +
        perceptual_weight * perceptual_loss +
        ssim_weight * ssim_loss +
        geometric_weight * geometric_loss
    )
    
    return {
        'total_loss': total_loss,
        'noise_loss': noise_loss,
        'latent_recon_loss': latent_recon_loss,
        'pixel_recon_loss': pixel_recon_loss,
        'perceptual_loss': perceptual_loss,
        'ssim_loss': ssim_loss,
        'ssim_value': ssim_value,
        'geometric_loss': geometric_loss,
        'decoded_images': decoded_images
    } 