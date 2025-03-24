from typing import Optional, Dict, Any, NamedTuple
from diffusers import UNet2DConditionModel
from .camera_encoder import CameraEncoder
from .pipeline import MVDPipeline
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class UNetOutput(NamedTuple):
    sample: torch.FloatTensor

class MultiViewUNet(nn.Module):
    def __init__(
        self, 
        pretrained_model_name_or_path,
        dtype: torch.dtype = torch.float32,
        use_memory_efficient_attention: bool = True,
        enable_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        use_memory_efficient_attention = True
        
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_memory_efficient_attention=use_memory_efficient_attention,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        self.config = self.base_unet.config        
        self.device = self.base_unet.device
        self.dtype  = self.base_unet.dtype

        if enable_gradient_checkpointing:
            self.base_unet.enable_gradient_checkpointing()
        
        self.camera_encoder = CameraEncoder(output_dim=1024).to(device=self.device, dtype=self.dtype)
        
        down_channels = self.config.block_out_channels
        up_channels   = list(reversed(down_channels))
        mid_channels  = down_channels[-1]
        
        num_down_blocks = len(self.base_unet.down_blocks)
        num_up_blocks   = len(self.base_unet.up_blocks)
        
        self.down_modulators = nn.ModuleList([
            nn.Linear(1024, down_channels[min(i, len(down_channels)-1)] * 2) 
            for i in range(num_down_blocks)
        ]).to(device=self.device, dtype=self.dtype)
        
        self.up_modulators = nn.ModuleList([
            nn.Linear(1024, up_channels[i] * 2)
            for i in range(num_up_blocks)
        ]).to(device=self.device, dtype=self.dtype)
        
        self.mid_modulator = nn.Linear(1024, mid_channels * 2).to(device=self.device, dtype=self.dtype)
        
        self.output_modulator = nn.Linear(1024, 4 * 2).to(device=self.device, dtype=self.dtype)


    def apply_modulation(self, hidden_states, modulator, camera_embedding):
        """Apply scale and shift modulation to hidden states"""
        # Generate scale and shift
        modulation = modulator(camera_embedding)  # [B, C*2]
        scale, shift = modulation.chunk(2, dim=-1)  # Each [B, C]
        
        # Reshape for broadcasting
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)  # [B, C, 1, 1]
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)  # [B, C, 1, 1]
        
        # Ensure scale and shift have the correct number of channels
        if scale.shape[1] != hidden_states.shape[1]:
            raise ValueError(f"Channel dimension mismatch: scale has {scale.shape[1]} channels but hidden states has {hidden_states.shape[1]} channels")
        
        return hidden_states * scale + shift


    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        
        self.device = device
        if 'dtype' in kwargs:
            self.dtype = dtype
            
        self.camera_encoder   = self.camera_encoder.to(device=device, dtype=dtype)
        self.down_modulators  = self.down_modulators.to(device=device, dtype=dtype)
        self.up_modulators    = self.up_modulators.to(device=device, dtype=dtype)
        self.mid_modulator    = self.mid_modulator.to(device=device, dtype=dtype)
        self.output_modulator = self.output_modulator.to(device=device, dtype=dtype)
        
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
        
        # Handle batch size mismatch between sample and encoder_hidden_states
        if sample.shape[0] > encoder_hidden_states.shape[0]:
            repeat_factor = sample.shape[0] // encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.repeat(repeat_factor, 1, 1)
        
        # Process camera information and generate camera embedding
        camera_embedding = self._process_camera(target_camera, sample.shape[0])
        
        # Run the base UNet
        output = self.base_unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        # Get output from base UNet
        hidden_states = output.sample
        
        # Apply camera modulation for output channels (should be 4 for latent space)
        if hidden_states.shape[1] == 4:
            hidden_states = self.apply_modulation(hidden_states, self.output_modulator, camera_embedding)
        
        if not return_dict:
            return hidden_states
            
        return UNetOutput(sample=hidden_states)
    
    
    def _process_camera(self, camera, batch_size):
        """
        Process camera information into embeddings.
        
        Parameters:
            camera: Camera information (dict, tensor, or None)
            batch_size: Batch size for creating default embeddings if needed
            
        Returns:
            camera_embedding: Tensor of shape [batch_size, 1024]
        """
        if camera is None:
            return torch.zeros(batch_size, 1024, device=self.device, dtype=self.dtype)
        
        if hasattr(camera, 'to'):
            camera = camera.to(device=self.device)
        
        try:
            # Convert camera format if needed
            if not isinstance(camera, dict) and hasattr(camera, 'shape') and camera.shape[-2:] == (4, 4):
                camera_dict = {
                    'R': camera[:, :3, :3].to(device=self.device),  # Rotation matrix [B, 3, 3]
                    'T': camera[:, :3, 3].to(device=self.device)    # Translation vector [B, 3]
                }
                return self.camera_encoder(camera_dict)
            elif isinstance(camera, dict):
                return self.camera_encoder(camera)
            else:
                # Unknown format, return zeros
                return torch.zeros(batch_size, 1024, device=self.device, dtype=self.dtype)
        except Exception as e:
            # Log error and return zeros
            logger.warning(f"Error processing camera: {str(e)}")
            return torch.zeros(batch_size, 1024, device=self.device, dtype=self.dtype)



def create_mvd_pipeline(
    pretrained_model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
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