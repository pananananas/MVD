import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from icecream import ic
import logging
import torch
from typing import Dict

logger = logging.getLogger(__name__)

class CameraEncoder(nn.Module):
    def __init__(self, output_dim: int = 768, max_freq: int = 10, modulation_hidden_dims: Dict[str, int] = None, modulation_strength: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.max_freq = max_freq
        
        self.pos_enc_dim = (output_dim // 2) // 3
        
        self.rotation_encoder = nn.Sequential(
            nn.Linear(9, 512),  # flattened rotation matrix
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
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.modulation_hidden_dims = modulation_hidden_dims or {}
        
        self.modulators = nn.ModuleDict()
        for name, dim in self.modulation_hidden_dims.items():
            self.modulators[name] = nn.Linear(output_dim, dim * 2)  # *2 for scale and shift
            
        self.init_modulators()
        
        self.current_step = 0
        self.total_steps = 10000
        self.modulation_strength = modulation_strength
        
        logger.info(f"CameraEncoder initialized with modulation_strength={modulation_strength}")
    

    def init_modulators(self):
        for name, modulator in self.modulators.items():
            nn.init.zeros_(modulator.weight)
            nn.init.zeros_(modulator.bias)
            
            # Set scale biases to 1.0 (first half) and shift biases to 0.0 (second half)
            dim = modulator.out_features // 2
            modulator.bias.data[:dim] = 1.0  # scale initialized to 1
                

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
            
        modulation = self.modulators[modulator_name](camera_embedding)
        scale, shift = modulation.chunk(2, dim=-1)
        
        # ic(f"{modulator_name}_scale", scale)
        # ic(f"{modulator_name}_shift", shift)

        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)

        scale = 1.0 + torch.tanh(scale) * 0.1
        
        before_stats = (tensor.min().item(), tensor.mean().item(), tensor.max().item())
        # ic(f"{modulator_name}_tensor_before_modulation", before_stats)
        
        modulated = tensor * (1.0 - self.modulation_strength + self.modulation_strength * scale) + self.modulation_strength * shift
        
        after_stats = (modulated.min().item(), modulated.mean().item(), modulated.max().item())
        # ic(f"{modulator_name}_tensor_after_modulation", after_stats)
        
        return modulated