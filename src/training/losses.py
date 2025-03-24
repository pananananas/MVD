from torchvision.transforms import Normalize
from torchvision.models import VGG16_Weights
import torchvision.models as models
import torch.nn.functional as F

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

def compute_geometric_consistency(generated, source, target):
    """
    Compute geometric consistency loss between views
    For latent representations, we use a simple L1 loss
    """
    return F.l1_loss(generated, target)

def compute_losses(noise_pred, noise, denoised_images, target_images, source_images, perceptual_loss_fn, ssim_loss_fn, config):
    """
    Compute all loss components for MVD training
    
    Args:
        noise_pred: Predicted noise (4-channel latent)
        noise: Ground truth noise (4-channel latent)
        denoised_images: Generated images after denoising (4-channel latent)
        target_images: Ground truth target view images (4-channel latent)
        source_images: Source view images (4-channel latent)
        perceptual_loss_fn: Instance of PerceptualLoss
        ssim_loss_fn: Instance of SSIM
        config: Training configuration with loss weights
    """
    # Debug logs to check loss functions
    print(f"perceptual_loss_fn: {perceptual_loss_fn}")
    print(f"ssim_loss_fn: {ssim_loss_fn}")
    print(f"Type of perceptual_loss_fn: {type(perceptual_loss_fn)}")
    print(f"Type of ssim_loss_fn: {type(ssim_loss_fn)}")
    
    noise_loss = F.mse_loss(noise_pred, noise)
    recon_loss = F.l1_loss(denoised_images, target_images)
    
    # SSIM and perceptual losses can't work on 4-channel latents
    # If needed, decode to pixel space first
    # For now, we'll just use simpler losses
    
    # Perceptual loss - skip for latent space
    if perceptual_loss_fn is None:
        print("WARNING: perceptual_loss_fn is None! Using a default loss instead.")
        perceptual_loss = recon_loss  # Use reconstruction loss as fallback
    else:
        print("WARNING: Perceptual loss not applicable in latent space, using recon_loss instead")
        perceptual_loss = recon_loss
    
    # Structural similarity loss - skip for latent space
    if ssim_loss_fn is None:
        print("WARNING: ssim_loss_fn is None! Using a default loss instead.")
        ssim_value = 0.0
        ssim_loss = 1.0
    else:
        print("WARNING: SSIM not applicable in latent space, using fixed value")
        ssim_value = 0.0
        ssim_loss = 1.0
    
    # Geometric consistency loss - works on latent space
    geometric_loss = compute_geometric_consistency(denoised_images, source_images, target_images)
    
    # Weight the loss components using config values
    total_loss = (
        1.0 * noise_loss +  # Make noise loss significant for latent training
        0.5 * recon_loss +  # Reconstruction in latent space
        config['perceptual_weight'] * perceptual_loss +
        config['ssim_weight']       * ssim_loss +
        config['geometric_weight']  * geometric_loss
    )
    
    return {
        'total_loss': total_loss,
        'noise_loss': noise_loss.item(),
        'recon_loss': recon_loss.item(),
        'perceptual_loss': perceptual_loss.item() if not isinstance(perceptual_loss, type(None)) else 0.0,
        'ssim_loss': ssim_loss.item() if not isinstance(ssim_loss, float) else ssim_loss,
        'geometric_loss': geometric_loss.item(),
        'ssim_value': ssim_value.item() if not isinstance(ssim_value, float) else ssim_value
    } 