from .attention import get_attention_processor_for_module
from typing import Optional, Dict, Any, NamedTuple
from diffusers import UNet2DConditionModel
from .camera_encoder import CameraEncoder
from .image_encoder import ImageEncoder
from .pipeline import MVDPipeline
import torch.nn as nn
import traceback
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
        
        self.image_encoder = ImageEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            dtype=dtype
        ).to(device=self.device, dtype=self.dtype)
        
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

        # Initialize attention processors for image cross-attention
        self._init_image_cross_attention()


    def _init_image_cross_attention(self):
        """
        Initialize image cross-attention processors for all attention layers in the UNet.
        This replaces the default attention processors with our custom ImageCrossAttentionProcessor.
        """
        logger.info("Initializing image cross-attention processors")
        
        # Dictionary to store attention layer mappings
        self.attention_layer_map = {}
        
        # Setup down blocks
        for i, block in enumerate(self.base_unet.down_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    for transformer_block in attn_block.transformer_blocks:
                        # Process attn1 (self-attention)
                        name = f"down_block_{i}_attn_{j}_self"
                        self._replace_attention_processor(transformer_block.attn1, name)
                        
                        # Process attn2 (cross-attention)
                        name = f"down_block_{i}_attn_{j}_cross"
                        self._replace_attention_processor(transformer_block.attn2, name)
        
        # Setup mid block
        if hasattr(self.base_unet.mid_block, 'attentions'):
            for j, attn_block in enumerate(self.base_unet.mid_block.attentions):
                for transformer_block in attn_block.transformer_blocks:
                    # Process attn1 (self-attention)
                    name = f"mid_block_attn_{j}_self"
                    self._replace_attention_processor(transformer_block.attn1, name)
                    
                    # Process attn2 (cross-attention)
                    name = f"mid_block_attn_{j}_cross"
                    self._replace_attention_processor(transformer_block.attn2, name)
        
        # Setup up blocks
        for i, block in enumerate(self.base_unet.up_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    for transformer_block in attn_block.transformer_blocks:
                        # Process attn1 (self-attention)
                        name = f"up_block_{i}_attn_{j}_self"
                        self._replace_attention_processor(transformer_block.attn1, name)
                        
                        # Process attn2 (cross-attention)
                        name = f"up_block_{i}_attn_{j}_cross"
                        self._replace_attention_processor(transformer_block.attn2, name)
        
        logger.info(f"Initialized image cross-attention for {len(self.attention_layer_map)} attention layers")
    
    def _replace_attention_processor(self, attn_module, name):
        """
        Replace the attention processor with our image cross-attention processor.
        
        Args:
            attn_module: Attention module to modify
            name: Unique identifier for this attention layer
        """
        # Get processor for this module
        processor = get_attention_processor_for_module(name, attn_module)
        
        # Save mapping from name to module for debugging
        self.attention_layer_map[name] = attn_module
        
        # Set the processor
        attn_module.processor = processor


    def apply_modulation(self, hidden_states, modulator, camera_embedding):
        """apply scale and shift modulation to hidden states"""
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
        self.image_encoder    = self.image_encoder.to(device=device, dtype=dtype)
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
        source_image_latents: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        timestep_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):

        sample = sample.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device)
        
        # the batch size mismatch happens when we use classifier-free guidance
        if sample.shape[0] > encoder_hidden_states.shape[0]:
            repeat_factor = sample.shape[0] // encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.repeat(repeat_factor, 1, 1)
        
        camera_embedding = self._process_camera(target_camera, sample.shape[0])
        
        # apply camera modulation for output channels (should be 4 for latent space)
        if sample.shape[1] == 4:
            sample = self.apply_modulation(sample, self.output_modulator, camera_embedding)
        
        # Initialize reference hidden states dictionary
        ref_hidden_states = None
        
        # process source image if provided
        if source_image_latents is not None:
            try:
                # use timestep 0 (beginning of diffusion) for feature extraction
                batch_size = source_image_latents.shape[0]
                encoder_timestep = torch.zeros(batch_size, device=self.device).long()
                
                image_encoder_text_embeddings = encoder_hidden_states
                
                # if we're using classifier-free guidance (embeddings are duplicated),
                # only use the conditional part (second half)
                if encoder_hidden_states.shape[0] == 2 * batch_size:
                    # take only the conditional embeddings (second half)
                    image_encoder_text_embeddings = encoder_hidden_states[batch_size:]
                elif encoder_hidden_states.shape[0] > batch_size:
                    # take only what we need to match the batch size
                    image_encoder_text_embeddings = encoder_hidden_states[:batch_size]
                
                # extract features from source image latents
                image_features = self.image_encoder(
                    latents=source_image_latents,
                    text_embeddings=image_encoder_text_embeddings,
                    timestep=encoder_timestep
                )
                
                feature_count = len(image_features)
                logger.info(f"Extracted features from source image: {feature_count} attention layers")
                
                # Map image features to UNet attention layers
                ref_hidden_states = self._map_image_features_to_attention_layers(image_features)
                
                # Log feature mapping for debugging
                logger.info(f"Mapped {len(ref_hidden_states)} image features to attention layers")
                
            except Exception as e:
                logger.warning(f"Error processing source image: {str(e)}")
                logger.warning(f"Error details: {e.__class__.__name__}")
                logger.warning(f"Source image latents shape: {source_image_latents.shape if source_image_latents is not None else None}")
                logger.warning(f"Encoder hidden states shape: {encoder_hidden_states.shape}")

                logger.warning(traceback.format_exc())
        
        # Update cross_attention_kwargs to include reference hidden states
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
            
        if ref_hidden_states is not None:
            cross_attention_kwargs["ref_hidden_states"] = ref_hidden_states
        
        # run the denoising UNet
        output = self.base_unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        # get the output from the denoising UNet
        hidden_states = output.sample
        
        if not return_dict:
            return hidden_states
            
        return UNetOutput(sample=hidden_states)
    
    
    # TODO: Improve that
    def _map_image_features_to_attention_layers(self, image_features):
        """
        Maps features extracted from the image encoder to the appropriate attention layers.
        
        Args:
            image_features: Dictionary of features from the image encoder
            
        Returns:
            Dictionary mapping attention layer names to corresponding image features
        """
        ref_hidden_states = {}
        
        # First, let's create a more structured mapping of our attention layers
        attention_layers_info = {
            "down": {},
            "mid": {},
            "up": {}
        }
        
        # Organize attention layers by block and level
        for name in self.attention_layer_map.keys():
            if name.startswith("down_block_"):
                parts = name.split("_")
                block_idx = int(parts[2])
                if block_idx not in attention_layers_info["down"]:
                    attention_layers_info["down"][block_idx] = []
                attention_layers_info["down"][block_idx].append(name)
            elif name.startswith("mid_block"):
                attention_layers_info["mid"].setdefault(0, []).append(name)
            elif name.startswith("up_block_"):
                parts = name.split("_")
                block_idx = int(parts[2])
                if block_idx not in attention_layers_info["up"]:
                    attention_layers_info["up"][block_idx] = []
                attention_layers_info["up"][block_idx].append(name)
        
        # Also organize image features by block and level
        image_features_info = {
            "down": {},
            "mid": {},
            "up": {}
        }
        
        for key, feature in image_features.items():
            if key.startswith("down_block_"):
                parts = key.split("_")
                block_idx = int(parts[2])
                if block_idx not in image_features_info["down"]:
                    image_features_info["down"][block_idx] = []
                image_features_info["down"][block_idx].append((key, feature))
            elif key.startswith("mid_block"):
                image_features_info["mid"].setdefault(0, []).append((key, feature))
            elif key.startswith("up_block_"):
                parts = key.split("_")
                block_idx = int(parts[2])
                if block_idx not in image_features_info["up"]:
                    image_features_info["up"][block_idx] = []
                image_features_info["up"][block_idx].append((key, feature))
        
        logger.info(f"Mapping features with semantically similar positions:")
        
        # Now let's match features to attention layers based on similar position in the network
        for section in ["down", "mid", "up"]:
            for block_idx in attention_layers_info[section]:
                if block_idx not in image_features_info[section]:
                    continue
                
                attn_layers = attention_layers_info[section][block_idx]
                features = image_features_info[section][block_idx]
                
                # If we have equal numbers, do a direct mapping
                if len(features) == len(attn_layers):
                    for i, (feat_name, feat) in enumerate(features):
                        layer_name = attn_layers[i]
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        ref_hidden_states[layer_name] = feat
                        logger.debug(f"Direct map: {feat_name} → {layer_name}")
                else:
                    # If numbers don't match, map features to most appropriate layers
                    # For now, distribute the features evenly across the layers
                    for i, layer_name in enumerate(attn_layers):
                        feat_idx = min(i % len(features), len(features) - 1)
                        _, feat = features[feat_idx]
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        ref_hidden_states[layer_name] = feat
                        logger.debug(f"Distributed map: feature {feat_idx} → {layer_name}")
        
        # Ensure all attention layers have a feature
        for name in self.attention_layer_map:
            if name not in ref_hidden_states:
                # Find the closest feature based on similar location in network
                if name.startswith("down_block_"):
                    parts = name.split("_")
                    block_idx = int(parts[2])
                    # Try to find a feature from the same block level
                    for search_idx in range(block_idx, -1, -1):
                        if search_idx in image_features_info["down"] and image_features_info["down"][search_idx]:
                            feat_name, feat = image_features_info["down"][search_idx][0]
                            if isinstance(feat, tuple):
                                feat = feat[0]
                            ref_hidden_states[name] = feat
                            logger.debug(f"Fallback map: {feat_name} → {name}")
                            break
                elif name.startswith("mid_block"):
                    # Try down_blocks last level or up_blocks first level
                    if image_features_info["mid"] and 0 in image_features_info["mid"]:
                        feat_name, feat = image_features_info["mid"][0][0]
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        ref_hidden_states[name] = feat
                        logger.debug(f"Fallback map: {feat_name} → {name}")
                elif name.startswith("up_block_"):
                    parts = name.split("_")
                    block_idx = int(parts[2])
                    # Try to find a feature from the same block level
                    for search_idx in range(block_idx, len(attention_layers_info["up"])):
                        if search_idx in image_features_info["up"] and image_features_info["up"][search_idx]:
                            feat_name, feat = image_features_info["up"][search_idx][0]
                            if isinstance(feat, tuple):
                                feat = feat[0]
                            ref_hidden_states[name] = feat
                            logger.debug(f"Fallback map: {feat_name} → {name}")
                            break
        
        # Count how many layers were successfully mapped
        logger.info(f"Mapped {len(ref_hidden_states)}/{len(self.attention_layer_map)} attention layers to image features")
        
        return ref_hidden_states
    
    
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