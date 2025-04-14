from torchvision.transforms import Normalize
from torchvision.models import VGG16_Weights
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import SSIM
import torch

class PerceptualLoss:
    def __init__(self, device='cuda'):
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:29].to(device).eval()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        """Move the model to the specified device"""
        self.device = device
        self.vgg = self.vgg.to(device)
        return self




def compute_losses(noise_pred, noise, denoised_latents=None, target_latents=None, vae=None, perceptual_loss_fn=None, ssim_loss_fn=None, config=None):
    
    noise_loss = F.mse_loss(noise_pred, noise)
    
    device = noise_loss.device
    zero_tensor = torch.tensor(0.0, device=device)
    
    return {
        'total_loss': noise_loss,
        'noise_loss': noise_loss,
        'latent_recon_loss': zero_tensor,
        'pixel_recon_loss': zero_tensor,
        'perceptual_loss': zero_tensor,
        'ssim_loss': zero_tensor,
        'ssim_value': zero_tensor,
        'geometric_loss': zero_tensor,
        'decoded_images': {'denoised': None, 'target': None}
    } 
