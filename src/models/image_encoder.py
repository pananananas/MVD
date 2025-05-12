from diffusers import UNet2DConditionModel
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class ImageEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        dtype: torch.dtype = torch.float32,
        expected_sample_size: int = None,
    ):
        super().__init__()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=dtype,
        )
        
        self.unet.config.sample_size = expected_sample_size
        
        for param in self.unet.parameters():
            param.requires_grad = False
            
        self.unet.eval()
        self.dtype = dtype
        self.device = "cpu"
        
        self.extracted_features = {}
        self._register_hooks()
        

    def _register_hooks(self):
        
        self.hooks = []
        registered_layers = set()
        
        for i, block in enumerate(self.unet.down_blocks):
            if hasattr(block, 'attentions'):
                logger.debug(f"Registering hooks for down_block_{i} (has {len(block.attentions)} attention layers)")
                for j, layer in enumerate(block.attentions):
                    name = f"down_block_{i}_attn_{j}"
                    if name not in registered_layers:
                        hook = layer.register_forward_hook(
                            lambda module, input, output, name=name: self._hook_fn(name, output)
                        )
                        self.hooks.append(hook)
                        registered_layers.add(name)
        
        for j, layer in enumerate(self.unet.mid_block.attentions):
            name = f"mid_block_attn_{j}"
            if name not in registered_layers:
                hook = layer.register_forward_hook(
                    lambda module, input, output, name=name: self._hook_fn(name, output)
                )
                self.hooks.append(hook)
                registered_layers.add(name)
        
        for i, block in enumerate(self.unet.up_blocks):
            if hasattr(block, 'attentions'):
                logger.debug(f"Registering hooks for up_block_{i} (has {len(block.attentions)} attention layers)")
                for j, layer in enumerate(block.attentions):
                    name = f"up_block_{i}_attn_{j}"
                    if name not in registered_layers:
                        hook = layer.register_forward_hook(
                            lambda module, input, output, name=name: self._hook_fn(name, output)
                        )
                        self.hooks.append(hook)
                        registered_layers.add(name)
                    
    
    def _hook_fn(self, name, output):
        self.extracted_features[name] = output[0] if isinstance(output, tuple) else output
                

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        
        self.device = device
        if 'dtype' in kwargs:
            self.dtype = dtype
            
        self.unet = self.unet.to(device=device, dtype=dtype)
        return super().to(*args, **kwargs)
    
    def forward(self, latents, text_embeddings, timestep):
        self.extracted_features = {}
        
        latents = latents.to(device=self.device, dtype=self.dtype)
        text_embeddings = text_embeddings.to(device=self.device)
        timestep = timestep.to(device=self.device)
        
        with torch.no_grad():
            self.unet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )
        
        return self.extracted_features