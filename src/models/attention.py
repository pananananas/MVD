from typing import Optional, Dict, Any
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class ImageCrossAttentionProcessor(nn.Module):
    def __init__(
        self,
        name: str,
        query_dim: int,
        heads: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        scale: float = 0.05,
    ):
        super().__init__()

        self.name = name
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.query_dim = query_dim
        
        self.original_processor = None
        
        # image cross-attention projections
        self.to_q_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        
        self.feature_adapter = None
        
        self.to_out_ref = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Dropout(dropout)
        ])
        
        self.ref_scale = nn.Parameter(torch.tensor(scale))
    
    def __call__(
        self,
        attn: Any,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        ref_hidden_states: Optional[Dict[str, torch.FloatTensor]] = None,
        ref_scale: float = 0.1,
        *args,
        **kwargs
    ) -> torch.FloatTensor:

        original_output = self.original_processor(
            attn, 
            hidden_states, 
            encoder_hidden_states, 
            attention_mask,
            temb=temb, 
            *args, 
            **kwargs
        )
        
        # Skip if no reference hidden states or no matching features for this layer
        if ref_hidden_states is None or self.name not in ref_hidden_states:
            return original_output
        
        # Get the reference features for this specific layer
        reference_states = ref_hidden_states[self.name]
        
        # Apply aggressive feature normalization to prevent instability
        with torch.no_grad():
            ref_mean = reference_states.mean().item()
            ref_std = reference_states.std().item()
            
            # Apply strong normalization to all reference features regardless of stats
            # Center at 0 and normalize to std=0.5
            reference_states = (reference_states - reference_states.mean(dim=(0, 1), keepdim=True)) 
            ref_std_tensor = torch.clamp(reference_states.std(dim=(0, 1), keepdim=True), min=1e-5)
            reference_states = reference_states / ref_std_tensor * 0.5
            
            # Log after normalization
            new_mean = reference_states.mean().item()
            new_std = reference_states.std().item()
            logger.info(f"Layer: {self.name} | Before norm: mean={ref_mean:.4f}, std={ref_std:.4f} | After: mean={new_mean:.4f}, std={new_std:.4f}")
        
        # Handle different input dimensions
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Adapt reference features if needed (dimensions only)
        reference_states = self._adapt_reference_features(reference_states, self.query_dim)
        
        # Skip if sequence lengths are extremely mismatched
        if reference_states.shape[1] > 10 * sequence_length or reference_states.shape[1] < sequence_length // 10:
            logger.warning(f"Sequence length mismatch in {self.name}: hidden={sequence_length}, ref={reference_states.shape[1]}")
            return original_output
        
        # Handle sequence length mismatch with simple interpolation
        if reference_states.shape[1] != sequence_length:
            logger.info(f"Interpolating reference features from {reference_states.shape[1]} to {sequence_length} tokens")
            # Use interpolation for sequence dimension
            reference_states = F.interpolate(
                reference_states.permute(0, 2, 1), 
                size=sequence_length,
                mode='linear'
            ).permute(0, 2, 1)
        
        # Step 1: Compute query from hidden states
        query = self.to_q_ref(hidden_states)
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Step 2: Compute key and value from reference features
        key = self.to_k_ref(reference_states)
        value = self.to_v_ref(reference_states)
        
        # Reshape key and value for attention
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Step 3: Compute attention 
        # Use scaled_dot_product_attention from PyTorch 2.0+ if available
        if hasattr(F, "scaled_dot_product_attention"):
            # This is more efficient on newer PyTorch versions
            ref_attention_output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=None,  # No mask for reference features
                dropout_p=0.0, 
                is_causal=False
            )

        ref_attention_output = ref_attention_output.transpose(1, 2).reshape(
            batch_size, -1, self.heads * self.dim_head
        )
        
        # Apply output projection
        ref_hidden_states = self.to_out_ref[0](ref_attention_output)
        ref_hidden_states = self.to_out_ref[1](ref_hidden_states)
        
        # Reshape back to input shape if needed
        if input_ndim == 4:
            ref_hidden_states = ref_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Monitor statistics of outputs before combination
        with torch.no_grad():
            orig_mean = original_output.mean().item()
            orig_std = original_output.std().item()
            ref_mean = ref_hidden_states.mean().item()
            ref_std = ref_hidden_states.std().item()
            
            logger.debug(f"Layer {self.name} - Original: mean={orig_mean:.4f}, std={orig_std:.4f} | " 
                         f"Reference: mean={ref_mean:.4f}, std={ref_std:.4f} | Scale: {ref_scale:.4f}")
            
            if abs(ref_mean) > 0.5 or ref_std > 1.5:
                logger.warning(f"Reference features have extreme values in {self.name}: mean={ref_mean:.4f}, std={ref_std:.4f}")
                # Apply normalization to prevent extreme values
                ref_hidden_states = ref_hidden_states / max(ref_std, 1.0)

        # Use a safer combination approach - clamp ref_scale to reasonable values
        safe_ref_scale = min(max(ref_scale, 0.0), 0.5)
        combined_output = original_output + safe_ref_scale * ref_hidden_states
        
        return combined_output
    
    def _adapt_reference_features(self, reference_states, target_dim):
        """
        Simplified adaptation that focuses on dimension matching only
        """
        # Get the shape of reference states
        if reference_states.ndim == 4:
            batch_size, channels, height, width = reference_states.shape
            # Reshape from [B, C, H, W] to [B, H*W, C]
            reference_states = reference_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Check if we need to adapt the feature dimension
        feature_dim = reference_states.shape[-1]
        if feature_dim != target_dim:
            # Create feature adapter if it doesn't exist or has wrong dimensions
            if self.feature_adapter is None or self.feature_adapter.in_features != feature_dim:
                self.feature_adapter = nn.Linear(feature_dim, target_dim).to(
                    device=reference_states.device, dtype=reference_states.dtype
                )
                # Initialize with identity-like mapping if possible
                if feature_dim <= target_dim:
                    with torch.no_grad():
                        self.feature_adapter.weight.zero_()
                        self.feature_adapter.bias.zero_()
                        # Set the overlapping part to identity
                        for i in range(min(feature_dim, target_dim)):
                            self.feature_adapter.weight[i, i] = 1.0
                else:
                    # Random init for dimension reduction
                    nn.init.xavier_uniform_(self.feature_adapter.weight)
                    nn.init.zeros_(self.feature_adapter.bias)
            
            # Apply adaptation
            reference_states = self.feature_adapter(reference_states)
            
        return reference_states
    
    def load_original_weights(self, attn_module):
        """
        Initialize the image cross-attention weights from the original attention module,
        maintaining the relationship between query, key, and value projections.
        
        Args:
            attn_module: Original attention module to copy weights from
        """
        with torch.no_grad():
            # Copy query weights (these should always match in dimension)
            self.to_q_ref.weight.copy_(attn_module.to_q.weight)
            
            # For output projection (should be safe to copy)
            self.to_out_ref[0].weight.copy_(attn_module.to_out[0].weight)
            self.to_out_ref[0].bias.copy_(attn_module.to_out[0].bias)
            
            # Get original key and value weights
            orig_k = attn_module.to_k.weight
            orig_v = attn_module.to_v.weight
            
            # Handle key weights
            k_out, k_in = self.to_k_ref.weight.shape
            ok_out, ok_in = orig_k.shape
            
            if k_out == ok_out and k_in == ok_in:
                # Direct copy if dimensions match
                self.to_k_ref.weight.copy_(orig_k)
            else:
                # Preserve pattern when dimensions don't match
                # For input dimension (columns)
                if k_in >= ok_in:
                    # Our input dim is larger - initialize first part with original weights
                    self.to_k_ref.weight[:, :ok_in].copy_(orig_k[:min(k_out, ok_out), :])
                    # Zero out the rest to minimize initial impact
                    if k_in > ok_in:
                        self.to_k_ref.weight[:, ok_in:].zero_()
                else:
                    # Our input dim is smaller - use a linear projection of original weights
                    projection = torch.nn.functional.linear(
                        torch.eye(k_in, device=orig_k.device), 
                        orig_k[:min(k_out, ok_out), :k_in]
                    )
                    self.to_k_ref.weight.copy_(projection)
            
            # Handle value weights with the same strategy
            v_out, v_in = self.to_v_ref.weight.shape
            ov_out, ov_in = orig_v.shape
            
            if v_out == ov_out and v_in == ov_in:
                # Direct copy if dimensions match
                self.to_v_ref.weight.copy_(orig_v)
            else:
                # Preserve pattern when dimensions don't match
                # For input dimension (columns)
                if v_in >= ov_in:
                    # Our input dim is larger - initialize first part with original weights
                    self.to_v_ref.weight[:, :ov_in].copy_(orig_v[:min(v_out, ov_out), :])
                    # Zero out the rest to minimize initial impact
                    if v_in > ov_in:
                        self.to_v_ref.weight[:, ov_in:].zero_()
                else:
                    # Our input dim is smaller - use a linear projection of original weights
                    projection = torch.nn.functional.linear(
                        torch.eye(v_in, device=orig_v.device), 
                        orig_v[:min(v_out, ov_out), :v_in]
                    )
                    self.to_v_ref.weight.copy_(projection)

def get_attention_processor_for_module(name, attn_module):
    """
    Create an ImageCrossAttentionProcessor for a given attention module.
    
    Args:
        name: Unique identifier for this processor
        attn_module: Attention module to create processor for
        
    Returns:
        Configured ImageCrossAttentionProcessor instance
    """
    # Get parameters from the attention module
    query_dim = attn_module.to_q.in_features
    heads = attn_module.heads
    dim_head = attn_module.to_q.out_features // heads
    
    # Create processor
    processor = ImageCrossAttentionProcessor(
        name=name,
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head
    )
    
    # Store original processor
    if hasattr(attn_module, 'processor'):
        processor.original_processor = attn_module.processor
    else:
        # If no processor exists, create a default processor that just calls the attention module
        def default_processor(attn, hidden_states, *args, **kwargs):
            return attn(hidden_states, *args, **kwargs)
        processor.original_processor = default_processor
    
    # Initialize the weights from the original attention module
    processor.load_original_weights(attn_module)
    
    logger.info(f"Created processor for {name} with query_dim={query_dim}, heads={heads}")
    
    return processor