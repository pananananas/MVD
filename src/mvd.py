from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any
from icecream import ic
import torch.nn as nn
import torch

class CameraEncoder(nn.Module):
    def __init__(self, output_dim: int = 768):  # 768 to standardowy wymiar embeddingu w SD
        super().__init__()
        # Wejście: R (9) + T (3) = 12 parametrów
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def to(self, *args, **kwargs):
        self.device = args[0] if args else kwargs.get('device', getattr(self, 'device', None))
        return super().to(*args, **kwargs)
    
    def forward(self, camera_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Ensure inputs are on the correct device
        R = camera_data['R'].to(self.device)
        T = camera_data['T'].to(self.device)
        
        # Spłaszcz macierz R i połącz z T
        R = R.view(R.shape[0], -1)  # [B, 9]
        camera_params = torch.cat([R, T], dim=1)  # [B, 12]
        
        return self.camera_encoder(camera_params)

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
        self.camera_encoder = CameraEncoder()
        
        # Get cross-attention dimension
        cross_attention_dim = self.base_unet.config.cross_attention_dim
        
        # Modify projection to handle the correct dimensions
        # The input will be concatenated text embeddings and camera embeddings
        self.projection = nn.Linear(cross_attention_dim + 2 * cross_attention_dim, cross_attention_dim)
        
        # Add dtype and device properties
        self.dtype = self.base_unet.dtype
        self.device = self.base_unet.device
    
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
        source_camera: Dict[str, torch.Tensor] = None,
        target_camera: Dict[str, torch.Tensor] = None,
        points_3d: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        timestep_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Move inputs to device
        sample = sample.to(self.device)
        timestep = timestep.to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.device)
        
        # During inference, we might not have camera parameters
        if source_camera is None or target_camera is None:
            # Just pass through to base UNet
            return self.base_unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict
            )
        
        B, S, D = encoder_hidden_states.shape
        # ic(encoder_hidden_states.shape)  # Should be [B, 77, 768]
        
        # Encode camera parameters
        source_camera_embedding = self.camera_encoder(source_camera)  # [B, 768]
        target_camera_embedding = self.camera_encoder(target_camera)  # [B, 768]
        # ic(source_camera_embedding.shape)  # Should be [B, 768]
        
        # Reshape camera embeddings to match text embeddings sequence length
        # Make sure we're using the correct batch size from encoder_hidden_states
        source_camera_embedding = source_camera_embedding.unsqueeze(1).expand(B, S, -1)
        target_camera_embedding = target_camera_embedding.unsqueeze(1).expand(B, S, -1)
        
        # ic(source_camera_embedding.shape)  # Should be [B, 77, 768]
        
        # Concatenate all embeddings along the feature dimension
        combined_embeddings = torch.cat([
            encoder_hidden_states,      # [B, S, D]
            source_camera_embedding,    # [B, S, D]
            target_camera_embedding,    # [B, S, D]
        ], dim=-1)  # Result: [B, S, 3*D]
        
        # ic(combined_embeddings.shape)  # Should be [B, 77, 2304]
        
        # Project back to original dimension
        enhanced_embeddings = self.projection(combined_embeddings)  # [B, S, D]
        # ic(enhanced_embeddings.shape)  # Should be [B, 77, 768]
        
        # Use the enhanced embeddings in base_unet
        return self.base_unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=enhanced_embeddings,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict
        )


def create_mvd_pipeline(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    dtype: torch.dtype = torch.float16,
    use_memory_efficient_attention: bool = True,
    enable_gradient_checkpointing: bool = True,
):
    # Load standard pipeline with only supported kwargs
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    
    # Replace UNet with our version
    mv_unet = MultiViewUNet(
        pretrained_model_name_or_path,
        dtype=dtype,
        use_memory_efficient_attention=use_memory_efficient_attention,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    pipeline.unet = mv_unet
    
    return pipeline