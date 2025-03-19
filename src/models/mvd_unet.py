from diffusers import UNet2DConditionModel
from .camera_encoder import CameraEncoder
from typing import Optional, Dict, Any
from .pipeline import MVDPipeline
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class MultiViewUNet(nn.Module):
    def __init__(
        self, 
        pretrained_model_name_or_path,
        dtype: torch.dtype = torch.float32,
        use_memory_efficient_attention: bool = True,
        enable_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        # Force memory efficient attention to reduce memory usage
        use_memory_efficient_attention = True
        
        # Load UNet with memory optimization
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_memory_efficient_attention=use_memory_efficient_attention,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        self.config = self.base_unet.config
        
        if enable_gradient_checkpointing:
            self.base_unet.enable_gradient_checkpointing()
        
        # Get device from base UNet for consistency
        self.device = self.base_unet.device
        self.dtype = self.base_unet.dtype
        
        # Create camera encoder
        self.camera_encoder = CameraEncoder(output_dim=1024).to(device=self.device, dtype=self.dtype)
        
        # In SD, the channel dimensions double at each downsampling step
        down_channels = self.config.block_out_channels
        up_channels   = list(reversed(down_channels))
        mid_channels  = down_channels[-1]
        
        num_down_blocks = len(self.base_unet.down_blocks)
        num_up_blocks   = len(self.base_unet.up_blocks)
        
        # Create modulation layers for each UNET block
        self.down_modulators = nn.ModuleList([
            nn.Linear(1024, down_channels[min(i, len(down_channels)-1)] * 2) 
            for i in range(num_down_blocks)
        ]).to(device=self.device, dtype=self.dtype)
        
        self.up_modulators = nn.ModuleList([
            nn.Linear(1024, up_channels[i] * 2)
            for i in range(num_up_blocks)
        ]).to(device=self.device, dtype=self.dtype)
        
        self.mid_modulator = nn.Linear(1024, mid_channels * 2).to(device=self.device, dtype=self.dtype)
        
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
        device = args[0] if args else kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        
        self.device = device
        if 'dtype' in kwargs:
            self.dtype = dtype
            
        # Ensure all components are moved to the correct device
        self.camera_encoder = self.camera_encoder.to(device=device, dtype=dtype)
        self.down_modulators = self.down_modulators.to(device=device, dtype=dtype)
        self.up_modulators = self.up_modulators.to(device=device, dtype=dtype)
        self.mid_modulator = self.mid_modulator.to(device=device, dtype=dtype)
        
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
        sample = sample.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device)
        
        target_camera = target_camera.to(device=self.device)
        
        target_camera_dict = {
            'R': target_camera[:, :3, :3].to(device=self.device),  # [B, 3, 3]
            'T': target_camera[:, :3, 3].to(device=self.device)    # [B, 3]
        }
        
        target_embedding = self.camera_encoder(target_camera_dict)  # [B, 1024]
        
        self.current_camera_embedding = target_embedding
        
        original_shape = sample.shape
        # resizing applied on macos to save ram
        max_size = 64
        
        if sample.shape[2] > max_size or sample.shape[3] > max_size:
            sample = F.interpolate(sample, size=(max_size, max_size), mode='bilinear')
        
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
            
            # If we resized the input, resize the output back to original shape
            if sample.shape != original_shape:
                if isinstance(output, dict) and 'sample' in output:
                    output['sample'] = F.interpolate(output['sample'], 
                                                size=(original_shape[2], original_shape[3]), 
                                                mode='bilinear')
                else:
                    output = F.interpolate(output, 
                                        size=(original_shape[2], original_shape[3]), 
                                        mode='bilinear')
            
            del self.current_camera_embedding
            return output
            

def create_mvd_pipeline(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    dtype: torch.dtype = torch.float16,  # Using float16 for memory efficiency
    use_memory_efficient_attention: bool = True,
    enable_gradient_checkpointing: bool = True,
    cache_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    pipeline = MVDPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    
    pipeline.safety_checker = None
    pipeline.feature_extractor = None
    
    mv_unet = MultiViewUNet(
        pretrained_model_name_or_path,
        dtype=dtype,
        use_memory_efficient_attention=use_memory_efficient_attention,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    mv_unet = mv_unet.to(device=device, dtype=dtype)
    pipeline.unet = mv_unet
    
    return pipeline