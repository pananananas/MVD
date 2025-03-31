from diffusers import UNet2DConditionModel
import torch.nn as nn
import torch
import logging

logger = logging.getLogger(__name__)

class ImageEncoder(nn.Module):
    """
    Extracts features from UNet attention layers when processing a source image.
    These features can be used for image-based conditioning in Multi-View Diffusion.
    """
    def __init__(
        self,
        pretrained_model_name_or_path,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=dtype,
        )
        
        # freeze UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False
            
        self.unet.eval()     # set to evaluation mode
        self.dtype = dtype
        self.device = "cpu"  # correct device set in forward
        
        self.extracted_features = {}
        self._register_hooks()
        

    def _register_hooks(self):
        """Register forward hooks to capture attention layer outputs"""
        
        self.hooks = []
        registered_layers = set()
        
        # down blocks
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
            else:
                logger.debug(f"Skipping down_block_{i} (no attention layers)")
        
        # mid block
        if hasattr(self.unet.mid_block, 'attentions'):
            logger.debug(f"Registering hooks for mid_block (has {len(self.unet.mid_block.attentions)} attention layers)")
            for j, layer in enumerate(self.unet.mid_block.attentions):
                name = f"mid_block_attn_{j}"
                if name not in registered_layers:
                    hook = layer.register_forward_hook(
                        lambda module, input, output, name=name: self._hook_fn(name, output)
                    )
                    self.hooks.append(hook)
                    registered_layers.add(name)
        else:
            logger.debug("Skipping mid_block (no attention layers)")
        
        # up blocks
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
            else:
                logger.debug(f"Skipping up_block_{i} (no attention layers)")
                    
        logger.info(f"Registered hooks for {len(registered_layers)} attention layers")
    
    def _hook_fn(self, name, output):
        """Store the output of the attention layer"""
        if isinstance(output, tuple):
            # Most likely the output is the attention output (features we want) 
            # followed by attention weights which we don't need
            self.extracted_features[name] = output[0]  # Take just the hidden states
        else:
            self.extracted_features[name] = output
            
        # Log the shape to help with debugging
        feat = self.extracted_features[name]
        logger.debug(f"Extracted feature '{name}' with shape {feat.shape}")
        
        # If it's a 3D tensor (B, L, D), make it 4D for consistency
        # This helps with dimension matching later
        if feat.ndim == 3:
            batch, seq_len, dim = feat.shape
            # Estimate a reasonable spatial shape (try to make it square-ish)
            side = int(seq_len**0.5)
            if side**2 == seq_len:  # Perfect square
                h, w = side, side
            else:
                # Find factors of seq_len that are closest to a square
                h, w = 1, seq_len
                for i in range(2, int(seq_len**0.5) + 1):
                    if seq_len % i == 0:
                        h, w = i, seq_len // i
                        # If we're getting closer to a square, update
                        if abs(h - w) < abs(h - seq_len // h):
                            h, w = i, seq_len // i
                        else:
                            break
            
            # Reshape to 4D for consistency across different types of features
            self.extracted_features[name] = feat.reshape(batch, h, w, dim).permute(0, 3, 1, 2)
            logger.debug(f"Reshaped feature '{name}' to 4D: {self.extracted_features[name].shape}")
    
    def to(self, *args, **kwargs):
        """Move the model to the specified device and dtype"""
        device = args[0] if args else kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.dtype)
        
        self.device = device
        if 'dtype' in kwargs:
            self.dtype = dtype
            
        self.unet = self.unet.to(device=device, dtype=dtype)
        return super().to(*args, **kwargs)
    
    def forward(self, latents, text_embeddings, timestep):
        """
        Extract features from the source image latents
        
        Args:
            latents: Encoded image latents from VAE [B, 4, H/8, W/8]
            text_embeddings: Text embeddings for conditioning [B, 77, 768]
            timestep: Current timestep in the diffusion process
            
        Returns:
            Dictionary of extracted features from attention layers
        """
        self.extracted_features = {}
        
        # move to the correct device
        latents         = latents.to(device=self.device, dtype=self.dtype)
        text_embeddings = text_embeddings.to(device=self.device)
        timestep        = timestep.to(device=self.device)
        
        # forward pass through UNet
        with torch.no_grad():
            self.unet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )
        
        return self.extracted_features