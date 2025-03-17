from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from .camera_encoder import CameraEncoder
from typing import Optional, Dict, Any
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class MultiViewUNet(nn.Module):
    def __init__(
        self, 
        pretrained_model_name_or_path,
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
        
        self.dtype = self.base_unet.dtype
        self.device = self.base_unet.device
        
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
        return_dict: bool = True,
        timestep_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Move inputs to device and cast to dtype
        sample = sample.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device)
        
        
        # Move camera data to device
        # source_camera = {k: v.to(self.device) for k, v in source_camera.items()}
        target_camera = {k: v.to(self.device) for k, v in target_camera.items()}
        
        # Encode camera parameters
        # source_embedding = self.camera_encoder(source_camera)  # [B, 1024]
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
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    
    pipeline.safety_checker = None
    pipeline.feature_extractor = None
    
    # print("\n\n\n\nBase pipeline.unet")
    # print(pipeline.unet)

    # Replace UNet with our version
    mv_unet = MultiViewUNet(
        pretrained_model_name_or_path,
        dtype=dtype,
        use_memory_efficient_attention=use_memory_efficient_attention,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    pipeline.unet = mv_unet
    # print("\n\n\n\nUpdated pipeline.unet")
    # print(pipeline.unet)
    
    return pipeline
