from .attention import get_attention_processor_for_module
from typing import Optional, Dict, Any, NamedTuple
from diffusers import UNet2DConditionModel
from .camera_encoder import CameraEncoder
from .image_encoder import ImageEncoder
from .pipeline import MVDPipeline
from icecream import ic
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
        img_ref_scale: float = 0.3,
        cam_modulation_strength: float = 0.2,
        use_camera_embeddings: bool = True,
        use_image_conditioning: bool = True,
    ):
        super().__init__()
        
        use_memory_efficient_attention = True
        self.use_camera_embeddings = use_camera_embeddings
        self.use_image_conditioning = use_image_conditioning
        
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
        self.img_ref_scale = img_ref_scale

        expected_sample_size = self.config.sample_size

        if enable_gradient_checkpointing:
            self.base_unet.enable_gradient_checkpointing()
        
        down_channels = self.config.block_out_channels
        up_channels   = list(reversed(down_channels))
        mid_channels  = down_channels[-1]
        
        num_down_blocks = len(self.base_unet.down_blocks)
        num_up_blocks   = len(self.base_unet.up_blocks)
        
        modulation_hidden_dims = {}
        for i in range(num_down_blocks):
            modulation_hidden_dims[f"down_{i}"] = down_channels[min(i, len(down_channels)-1)]
        
        for i in range(num_up_blocks):
            modulation_hidden_dims[f"up_{i}"] = up_channels[i]
        
        modulation_hidden_dims["mid"] = mid_channels
        modulation_hidden_dims["output"] = 4

        if self.use_camera_embeddings:
            self.camera_encoder = CameraEncoder(
                output_dim=1024, 
                modulation_hidden_dims=modulation_hidden_dims,
                modulation_strength=cam_modulation_strength
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.camera_encoder = None

        if self.use_image_conditioning:
            self.image_encoder = ImageEncoder(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                dtype=dtype,
                expected_sample_size=expected_sample_size
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.image_encoder = None
        
        self.hooks = []
        
        self._init_image_cross_attention()

    def _init_image_cross_attention(self):
        self.attention_layer_map = {}
        self.feature_to_attention_map = {}
        
        for i, block in enumerate(self.base_unet.down_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    for transformer_block in attn_block.transformer_blocks:
                        name = f"down_block_{i}_attn_{j}_self"
                        self._replace_attention_processor(transformer_block.attn1, name)
                        feature_name = f"down_block_{i}_attn_{j}"
                        self.feature_to_attention_map[feature_name] = [name]
                        
                        name = f"down_block_{i}_attn_{j}_cross"
                        self._replace_attention_processor(transformer_block.attn2, name)
                        if feature_name in self.feature_to_attention_map:
                            self.feature_to_attention_map[feature_name].append(name)
                        else:
                            self.feature_to_attention_map[feature_name] = [name]
        
        if hasattr(self.base_unet.mid_block, 'attentions'):
            for j, attn_block in enumerate(self.base_unet.mid_block.attentions):
                for transformer_block in attn_block.transformer_blocks:
                    name = f"mid_block_attn_{j}_self"
                    self._replace_attention_processor(transformer_block.attn1, name)
                    feature_name = f"mid_block_attn_{j}"
                    self.feature_to_attention_map[feature_name] = [name]
                    
                    name = f"mid_block_attn_{j}_cross"
                    self._replace_attention_processor(transformer_block.attn2, name)
                    if feature_name in self.feature_to_attention_map:
                        self.feature_to_attention_map[feature_name].append(name)
                    else:
                        self.feature_to_attention_map[feature_name] = [name]
        
        for i, block in enumerate(self.base_unet.up_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    for transformer_block in attn_block.transformer_blocks:
                        name = f"up_block_{i}_attn_{j}_self"
                        self._replace_attention_processor(transformer_block.attn1, name)
                        feature_name = f"up_block_{i}_attn_{j}"
                        self.feature_to_attention_map[feature_name] = [name]
                        
                        name = f"up_block_{i}_attn_{j}_cross"
                        self._replace_attention_processor(transformer_block.attn2, name)
                        if feature_name in self.feature_to_attention_map:
                            self.feature_to_attention_map[feature_name].append(name)
                        else:
                            self.feature_to_attention_map[feature_name] = [name]
        

    def _replace_attention_processor(self, attn_module, name):
        processor = get_attention_processor_for_module(name, attn_module, img_ref_scale=self.img_ref_scale)
        self.attention_layer_map[name] = attn_module
        attn_module.processor = processor


    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        
        self.device = device
        if 'dtype' in kwargs:
            self.dtype = dtype
            
        if self.camera_encoder is not None: 
            self.camera_encoder = self.camera_encoder.to(device=device, dtype=dtype)
        if self.image_encoder is not None:
            self.image_encoder = self.image_encoder.to(device=device, dtype=dtype)
        
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
        
        if sample.shape[0] > encoder_hidden_states.shape[0]:   # for classifier-free guidance
            repeat_factor = sample.shape[0] // encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.repeat(repeat_factor, 1, 1)
        
        self.current_camera_embedding = None # reset embedding
        
        if self.use_camera_embeddings and target_camera is not None:
            self._manage_modulation_hooks(register=True)
            self.current_camera_embedding = self.camera_encoder.encode_cameras(source_camera, target_camera)

            if self.current_camera_embedding is not None:
                sample = self.camera_encoder.apply_modulation(sample, "output", self.current_camera_embedding)
        # else:
            #  self._manage_modulation_hooks(register=False)

        ref_hidden_states = None
        
        if self.use_image_conditioning and source_image_latents is not None:
            
            batch_size = source_image_latents.shape[0]
            encoder_timestep = torch.zeros(batch_size, device=self.device).long()
            
            image_encoder_text_embeddings = encoder_hidden_states
            
            if encoder_hidden_states.shape[0] == 2 * batch_size:  # for classifier-free guidance
                image_encoder_text_embeddings = encoder_hidden_states[batch_size:]
            elif encoder_hidden_states.shape[0] > batch_size:
                image_encoder_text_embeddings = encoder_hidden_states[:batch_size]
            
            image_features = self.image_encoder(
                latents=source_image_latents,
                text_embeddings=image_encoder_text_embeddings,
                timestep=encoder_timestep
            )
            
            ref_hidden_states = self._map_image_features_to_attention_layers(image_features)
        
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
            
        if ref_hidden_states is not None:
            cross_attention_kwargs["ref_hidden_states"] = ref_hidden_states
        
        output = self.base_unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        hidden_states = output.sample
        
        if not return_dict:
            return hidden_states
            
        return UNetOutput(sample=hidden_states)
    
    
    def _map_image_features_to_attention_layers(self, image_features):
        ref_hidden_states = {}
        
        for name, feature in image_features.items():
            if isinstance(feature, tuple):
                feature = feature[0]
            
            if name in self.feature_to_attention_map:
                for attn_name in self.feature_to_attention_map[name]:
                    if attn_name in self.attention_layer_map:
                        ref_hidden_states[attn_name] = feature
        
        return ref_hidden_states
        

    def _manage_modulation_hooks(self, register: bool):
        if register and not self.hooks:

            def get_hook(idx, direction):
                def hook(module, inputs, output):
                    if hasattr(self, 'current_camera_embedding') and self.current_camera_embedding is not None:
                        return self.camera_encoder.apply_modulation(output, f"{direction}_{idx}", self.current_camera_embedding)
                    return output
                return hook

            # save places to apply hooks (cam modulation)
            hook_targets = []
            for i, block in enumerate(self.base_unet.down_blocks):
                hook_targets.append((block, get_hook(i, "down")))
            hook_targets.append((self.base_unet.mid_block, get_hook(0, "mid")))
            for i, block in enumerate(self.base_unet.up_blocks):
                hook_targets.append((block, get_hook(i, "up")))

            # apply hooks
            for module, hook_func in hook_targets:
                self.hooks.append(module.register_forward_hook(hook_func))

        elif not register and self.hooks:
            for handle in self.hooks:
                handle.remove()
            self.hooks = []


def create_mvd_pipeline(
    pretrained_model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
    use_memory_efficient_attention: bool = True,
    enable_gradient_checkpointing: bool = True,
    use_camera_embeddings: bool = True,
    use_image_conditioning: bool = True,
    img_ref_scale: float = 0.3,
    cam_modulation_strength: float = 0.2,
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
        img_ref_scale=img_ref_scale,
        cam_modulation_strength=cam_modulation_strength,
        use_camera_embeddings=use_camera_embeddings,
        use_image_conditioning=use_image_conditioning,
    )
    mv_unet = mv_unet.to(device=device, dtype=dtype)
    pipeline.unet = mv_unet
    
    pipeline.use_camera_embeddings = use_camera_embeddings
    pipeline.use_image_conditioning = use_image_conditioning
    pipeline.img_ref_scale = img_ref_scale
    
    return pipeline