import torch.nn.functional as F
from icecream import ic
from typing import Dict
import torch.nn as nn
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

class CameraEncoder(nn.Module):
    def __init__(self, output_dim: int = 768, max_freq: int = 10, modulation_hidden_dims: Dict[str, int] = None, modulation_strength: float = 1.0):
        super().__init__()
        self.output_dim = output_dim
        self.max_freq = max_freq
        
        self.pos_enc_dim = (output_dim // 2) // 3
        
        self.rotation_encoder = nn.Sequential(
            nn.Linear(9, 512),  # flattened rotation matrix
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )
        
        self.translation_encoder = nn.Sequential(
            nn.Linear(output_dim, 512),  
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.modulation_hidden_dims = modulation_hidden_dims or {}
        
        self.modulators = nn.ModuleDict()
        for name, dim in self.modulation_hidden_dims.items():
            self.modulators[name] = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.LayerNorm(output_dim // 2),
                nn.SiLU(),
                nn.Linear(output_dim // 2, dim * 2)  # *2 for scale and shift
            )
            
        self.init_modulators()
        
        self.modulation_strength = modulation_strength
        
        self._current_modulation_stats = {}

        logger.info(f"CameraEncoder initialized with modulation_strength={modulation_strength}")
    

    def init_modulators(self):
        for name, modulator in self.modulators.items():
            if isinstance(modulator, nn.Sequential):
                final_layer = modulator[-1]
            else:
                final_layer = modulator
                
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.02)
            
            dim = final_layer.out_features // 2
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                final_layer.bias.data[:dim].fill_(0.5)
                final_layer.bias.data[dim:].fill_(0.0)

    def compute_relative_transform(self, source_camera: torch.Tensor, target_camera: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        source_R = source_camera[:, :3, :3] 
        source_T = source_camera[:, :3, 3]
        target_R = target_camera[:, :3, :3]
        target_T = target_camera[:, :3, 3]
        
        source_to_target_R = torch.bmm(target_R, source_R.transpose(1, 2))            
        source_to_target_T = target_T - torch.bmm(source_to_target_R, source_T.unsqueeze(2)).squeeze(2)
        
        return {
            'R': source_to_target_R,
            'T': source_to_target_T
        }


    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        
        if device is not None:
            self.rotation_encoder = self.rotation_encoder.to(device=device, dtype=dtype)
            self.translation_encoder = self.translation_encoder.to(device=device, dtype=dtype)
            self.final_projection = self.final_projection.to(device=device, dtype=dtype)
            self.output_norm = self.output_norm.to(device=device, dtype=dtype)
            self.modulators = self.modulators.to(device=device, dtype=dtype)
        
        return super().to(*args, **kwargs)
    

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:

        device = x.device
        batch_size = x.shape[0]
        freqs = torch.exp(torch.linspace(0., np.log(self.max_freq), self.pos_enc_dim, device=device))
        
        x_expanded = x.unsqueeze(-1)
        angles = x_expanded * freqs[None, None, :]
        
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        encoding = encoding.reshape(batch_size, -1)
        
        weight = torch.randn(self.output_dim, encoding.shape[-1], device=device) / np.sqrt(encoding.shape[-1])
        encoding = F.linear(encoding, weight)
        
        return encoding
    

    def encode_cameras(self, source_camera: torch.Tensor, target_camera: torch.Tensor) -> torch.Tensor:

        device = next(self.parameters()).device
        if hasattr(source_camera, 'to'):
            source_camera = source_camera.to(device)
        if hasattr(target_camera, 'to'):
            target_camera = target_camera.to(device)
        
        camera_data = self.compute_relative_transform(source_camera, target_camera)
        
        return self.forward(camera_data)
    

    def forward(self, camera_data: Dict[str, torch.Tensor]) -> torch.Tensor:

        R = camera_data['R']
        T = camera_data['T']
        
        device = R.device
        
        if next(self.rotation_encoder.parameters()).device != device:
            self.to(device)
        
        R = R.to(device)
        T = T.to(device)
        
        R_flat = R.reshape(R.shape[0], -1)
        rotation_embedding = self.rotation_encoder(R_flat)
        
        translation_encoded = self.positional_encoding(T)
        translation_embedding = self.translation_encoder(translation_encoded)
        
        combined = torch.cat([rotation_embedding, translation_embedding], dim=-1)
        camera_embedding = self.final_projection(combined)

        camera_embedding = self.output_norm(camera_embedding)
        
        return camera_embedding
    
    
    def apply_modulation(self, hidden_states, modulator_name: str, camera_embedding: torch.Tensor):
        if isinstance(hidden_states, tuple):
            modulated_main = self.apply_modulation_to_tensor(hidden_states[0], modulator_name, camera_embedding)
            return (modulated_main,) + hidden_states[1:]
        else:
            return self.apply_modulation_to_tensor(hidden_states, modulator_name, camera_embedding)


    def apply_modulation_to_tensor(self, tensor, modulator_name, camera_embedding):
        if modulator_name not in self.modulators:
            return tensor
        
        if camera_embedding is None:
            ic(f"WARNING: apply_modulation called for {modulator_name} but camera_embedding is None. Skipping modulation.")
            return tensor

        modulation = self.modulators[modulator_name](camera_embedding)
        scale, shift = modulation.chunk(2, dim=-1)

        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)

        scale_processed = torch.sigmoid(scale) * 2.0 * self.modulation_strength
        shift_processed = shift * self.modulation_strength

        with torch.no_grad():
            before_mean = tensor.mean().item()
            before_std = tensor.std().item()

        modulated = tensor * scale_processed + shift_processed

        with torch.no_grad():

            after_mean = modulated.mean().item()
            after_std = modulated.std().item()
            scale_mean = scale_processed.mean().item()
            scale_std = scale_processed.std().item()
            shift_mean = shift_processed.mean().item()
            shift_std = shift_processed.std().item()

        self._current_modulation_stats[modulator_name] = {
            "before_mean": before_mean,
            "before_std": before_std,
            "scale_mean": scale_mean,
            "scale_std": scale_std,
            "shift_mean": shift_mean,
            "shift_std": shift_std,
            "after_mean": after_mean,
            "after_std": after_std,
        }


        return modulated