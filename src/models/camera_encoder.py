import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class CameraEncoder(nn.Module):
    def __init__(self, output_dim: int = 768, max_freq: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.max_freq = max_freq
        
        self.pos_enc_dim = (output_dim // 2) // 3  # Divide by 3 for x,y,z and by 2 for sin/cos
        
        self.rotation_encoder = nn.Sequential(
            nn.Linear(9, 512),  # Flattened rotation matrix
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self.translation_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    

    def to(self, *args, **kwargs):
        """Override to() to ensure all submodules are moved to the correct device"""
        device = args[0] if args else kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        
        if device is not None:
            self.rotation_encoder = self.rotation_encoder.to(device=device, dtype=dtype)
            self.translation_encoder = self.translation_encoder.to(device=device, dtype=dtype)
            self.final_projection = self.final_projection.to(device=device, dtype=dtype)
        
        return super().to(*args, **kwargs)
    

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for translation vectors
        Args:
            x: tensor of shape [B, 3] containing translation vectors
        Returns:
            encoding of shape [B, output_dim]
        """
        device = x.device
        batch_size = x.shape[0]
        freqs = torch.exp(torch.linspace(0., np.log(self.max_freq), self.pos_enc_dim, device=device))
        
        x_expanded = x.unsqueeze(-1)  # [B, 3, 1]
        angles = x_expanded * freqs[None, None, :]  # [B, 3, pos_enc_dim]
        
        sin_enc = torch.sin(angles)  # [B, 3, pos_enc_dim]
        cos_enc = torch.cos(angles)  # [B, 3, pos_enc_dim]
        
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, 3, 2*pos_enc_dim]
        encoding = encoding.reshape(batch_size, -1)       # [B, 6*pos_enc_dim]
        
        weight = torch.randn(self.output_dim, encoding.shape[-1], device=device) / np.sqrt(encoding.shape[-1])
        encoding = F.linear(encoding, weight)
        
        return encoding
    
    def forward(self, camera_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process camera parameters and return combined embedding
        Args:
            camera_data: dict containing 'R' [B, 3, 3] and 'T' [B, 3]
        Returns:
            camera_embedding: [B, output_dim]
        """
        R = camera_data['R']
        T = camera_data['T']
        
        device = R.device
        
        if next(self.rotation_encoder.parameters()).device != device:
            self.rotation_encoder = self.rotation_encoder.to(device)
            self.translation_encoder = self.translation_encoder.to(device)
            self.final_projection = self.final_projection.to(device)
        
        R = R.to(device)
        T = T.to(device)
        
        R_flat = R.reshape(R.shape[0], -1)
        rotation_embedding = self.rotation_encoder(R_flat)
        
        translation_encoded = self.positional_encoding(T)
        translation_embedding = self.translation_encoder(translation_encoded)
        
        combined = torch.cat([rotation_embedding, translation_embedding], dim=-1)
        camera_embedding = self.final_projection(combined)
        
        return camera_embedding