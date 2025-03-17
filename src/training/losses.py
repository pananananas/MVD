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
    For now, a simple L1 loss between generated and target images
    TODO: Implement more sophisticated geometric consistency check
    """
    return F.l1_loss(generated, target)

def compute_losses(noise_pred, noise, denoised_images, target_images, source_images, perceptual_loss_fn, ssim_loss_fn, config):
    """
    Compute all loss components for MVD training
    
    Args:
        noise_pred: Predicted noise
        noise: Ground truth noise
        denoised_images: Generated images after denoising
        target_images: Ground truth target view images
        source_images: Source view images
        perceptual_loss_fn: Instance of PerceptualLoss
        ssim_loss_fn: Instance of SSIM
        config: Training configuration with loss weights
    """
    # Basic losses
    noise_loss = F.mse_loss(noise_pred[:, :3], noise)
    recon_loss = F.l1_loss(denoised_images[:, :3], target_images)
    
    # Perceptual loss
    perceptual_loss = perceptual_loss_fn(denoised_images[:, :3], target_images)
    
    # Structural similarity loss
    ssim_value = ssim_loss_fn(denoised_images[:, :3], target_images)
    ssim_loss = 1 - ssim_value
    
    # Geometric consistency loss
    geometric_loss = compute_geometric_consistency(denoised_images[:, :3], source_images, target_images)
    
    # Weight the loss components using config values
    total_loss = (
        # 1.0 * noise_loss +
        # 1.0 * recon_loss +
        config['perceptual_weight'] * perceptual_loss +
        config['ssim_weight']       * ssim_loss +
        config['geometric_weight']  * geometric_loss
    )
    
    return {
        'total_loss': total_loss,
        'noise_loss': noise_loss.item(),
        'recon_loss': recon_loss.item(),
        'perceptual_loss': perceptual_loss.item(),
        'ssim_loss': ssim_loss.item(),
        'geometric_loss': geometric_loss.item(),
        'ssim_value': ssim_value.item()
    } 