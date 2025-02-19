from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class CameraEncoder(nn.Module):
    def __init__(self, output_dim: int = 768, max_freq: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.max_freq = max_freq
        
        # Calculate positional encoding dimension
        self.pos_enc_dim = (output_dim // 2) // 3  # Divide by 3 for x,y,z and by 2 for sin/cos
        
        # Separate encoders for rotation and translation
        self.rotation_encoder = nn.Sequential(
            nn.Linear(9, 512),  # Flattened rotation matrix
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self.translation_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),  # Takes positional encoded translation
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for translation vectors
        Args:
            x: tensor of shape [B, 3] containing translation vectors
        Returns:
            encoding of shape [B, output_dim]
        """
        batch_size = x.shape[0]
        freqs = torch.exp(torch.linspace(0., np.log(self.max_freq), self.pos_enc_dim, device=x.device))
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # [B, 3, 1]
        angles = x_expanded * freqs[None, None, :]  # [B, 3, pos_enc_dim]
        
        # Calculate sin and cos
        sin_enc = torch.sin(angles)  # [B, 3, pos_enc_dim]
        cos_enc = torch.cos(angles)  # [B, 3, pos_enc_dim]
        
        # Combine and reshape
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, 3, 2*pos_enc_dim]
        encoding = encoding.reshape(batch_size, -1)  # [B, 6*pos_enc_dim]
        
        # Project to desired output dimension
        encoding = F.linear(
            encoding,
            torch.randn(self.output_dim, encoding.shape[-1], device=x.device) / np.sqrt(encoding.shape[-1])
        )
        
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
        
        # Process rotation matrix
        R_flat = R.reshape(R.shape[0], -1)  # [B, 9]
        rotation_embedding = self.rotation_encoder(R_flat)
        
        # Process translation with positional encoding
        translation_encoded = self.positional_encoding(T)  # [B, output_dim]
        translation_embedding = self.translation_encoder(translation_encoded)
        
        # Combine embeddings
        combined = torch.cat([rotation_embedding, translation_embedding], dim=-1)
        camera_embedding = self.final_projection(combined)
        
        return camera_embedding

class MultiViewUNet(nn.Module):
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        dtype: torch.dtype = torch.float16,
        use_memory_efficient_attention: bool = True,
        enable_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        # Load base UNet
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_memory_efficient_attention=use_memory_efficient_attention,
            torch_dtype=dtype
        )
        
        # Expose config from base_unet
        self.config = self.base_unet.config
        
        if enable_gradient_checkpointing:
            self.base_unet.enable_gradient_checkpointing()
        
        # Add camera encoder
        self.camera_encoder = CameraEncoder(output_dim=1024)  # Fixed size for positional encodings
        
        # Get channel dimensions from UNet config
        # In Stable Diffusion, the channel dimensions double at each downsampling step
        # Starting from block_out_channels[0]
        down_channels = self.config.block_out_channels
        up_channels = list(reversed(down_channels))  # Include all channels for up blocks
        mid_channels = down_channels[-1]  # Mid block has same channels as deepest down block
        
        # Count actual number of blocks
        num_down_blocks = len(self.base_unet.down_blocks)
        num_up_blocks = len(self.base_unet.up_blocks)
        
        # Create modulation layers for each resolution
        self.down_modulators = nn.ModuleList([
            nn.Linear(1024, down_channels[min(i, len(down_channels)-1)] * 2) 
            for i in range(num_down_blocks)
        ])
        self.up_modulators = nn.ModuleList([
            nn.Linear(1024, up_channels[i] * 2)  # Use exact index since we have all channels
            for i in range(num_up_blocks)
        ])
        self.mid_modulator = nn.Linear(1024, mid_channels * 2)
        
        # Add dtype and device properties
        self.dtype = self.base_unet.dtype
        self.device = self.base_unet.device
        
        # Hook the UNet blocks to apply modulation
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on UNet blocks to apply modulation"""
        def get_modulation_hook(modulator):
            def hook(module, input, output):
                # Get the current camera embedding from the stored state
                if not hasattr(self, 'current_camera_embedding'):
                    return output
                
                # Handle tuple output case (some UNet blocks return multiple tensors)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Generate scale and shift
                modulation = modulator(self.current_camera_embedding)  # [B, C*2]
                scale, shift = modulation.chunk(2, dim=-1)  # Each [B, C]
                
                # Reshape for broadcasting
                scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)  # [B, C, 1, 1]
                shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)  # [B, C, 1, 1]
                
                # Ensure scale and shift have the correct number of channels
                if scale.shape[1] != hidden_states.shape[1]:
                    raise ValueError(f"Channel dimension mismatch: scale has {scale.shape[1]} channels but hidden states has {hidden_states.shape[1]} channels")
                
                # Apply modulation
                modulated_states = hidden_states * scale + shift
                
                # Return in the same format as input
                if isinstance(output, tuple):
                    return (modulated_states,) + output[1:]
                return modulated_states
            
            return hook
        
        # Register hooks for down blocks
        for idx, block in enumerate(self.base_unet.down_blocks):
            block.register_forward_hook(get_modulation_hook(self.down_modulators[idx]))
        
        # Register hook for mid block
        self.base_unet.mid_block.register_forward_hook(get_modulation_hook(self.mid_modulator))
        
        # Register hooks for up blocks
        for idx, block in enumerate(self.base_unet.up_blocks):
            block.register_forward_hook(get_modulation_hook(self.up_modulators[idx]))
    
    def to(self, *args, **kwargs):
        self.device = args[0] if args else kwargs.get('device', self.device)
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        self.camera_encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        source_camera: Optional[Dict[str, torch.Tensor]] = None,
        target_camera: Optional[Dict[str, torch.Tensor]] = None,
        points_3d: Optional[torch.FloatTensor] = None,  # Keep for backward compatibility
        return_dict: bool = True,
        timestep_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Move inputs to device and cast to dtype
        sample = sample.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device)
        
        # During inference, we might not have camera parameters
        if source_camera is None or target_camera is None:
            return self.base_unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict
            )
        
        # Move camera data to device
        source_camera = {k: v.to(self.device) for k, v in source_camera.items()}
        target_camera = {k: v.to(self.device) for k, v in target_camera.items()}
        
        # Encode camera parameters
        source_embedding = self.camera_encoder(source_camera)  # [B, 1024]
        target_embedding = self.camera_encoder(target_camera)  # [B, 1024]
        
        # Store the camera embedding for use in the hooks
        # We use target embedding as we want to condition on the target view
        self.current_camera_embedding = target_embedding
        
        # Forward through base UNet
        output = self.base_unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict
        )
        
        # Clean up the stored embedding
        del self.current_camera_embedding
        
        return output

def create_mvd_pipeline(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    dtype: torch.dtype = torch.float16,
    use_memory_efficient_attention: bool = True,
    enable_gradient_checkpointing: bool = True,
    cache_dir=None,
):
    # Initialize the pipeline with only the relevant parameters
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    
    # Disable safety checker
    pipeline.safety_checker = None
    pipeline.feature_extractor = None
    
    # Replace UNet with our version
    mv_unet = MultiViewUNet(
        pretrained_model_name_or_path,
        dtype=dtype,
        use_memory_efficient_attention=use_memory_efficient_attention,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    pipeline.unet = mv_unet
    
    return pipeline