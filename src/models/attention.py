from typing import Optional, Dict, Any
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)

class ImageCrossAttentionProcessor(nn.Module):
    """
    Attention processor that adds image-based cross-attention to the standard attention mechanism.
    
    This processor preserves the original attention computation and adds a parallel
    path for image cross-attention, combining the results with a learnable scale factor.
    """
    def __init__(
        self,
        name: str,
        query_dim: int,
        heads: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        scale: float = 0.1,
    ):
        """
        Args:
            name: Unique identifier for this processor (for feature mapping)
            query_dim: Dimension of the query vectors
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            dropout: Dropout probability
            scale: Initial scale factor for image cross-attention contribution
        """
        super().__init__()

        self.name = name
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.query_dim = query_dim
        
        # Store original processor
        self.original_processor = None
        
        # Image cross-attention projections
        self.to_q_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v_ref = nn.Linear(query_dim, self.inner_dim, bias=False)
        
        # Feature dimension adapter - will be initialized if needed
        self.feature_adapter = None
        
        self.to_out_ref = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Dropout(dropout)
        ])
        
        # Learnable scale parameter to control the contribution of image cross-attention
        self.ref_scale = nn.Parameter(torch.tensor(scale))
    
    def __call__(
        self,
        attn: Any,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        ref_hidden_states: Optional[Dict[str, torch.FloatTensor]] = None,
        *args,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Process attention with additional image cross-attention.
        
        Args:
            attn: Attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Text encoder hidden states (for regular cross-attention)
            attention_mask: Attention mask
            temb: Time embedding (not used here)
            ref_hidden_states: Dictionary mapping layer names to reference image features
            
        Returns:
            Combined attention output from both original and image cross-attention paths
        """
        # First run the original attention (without modifying it)
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
        
        # Handle different input dimensions
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Adapt reference features if needed
        reference_states = self._adapt_reference_features(reference_states, self.query_dim)
            
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
        else:
            # Fallback for older PyTorch versions
            scale = 1 / (self.dim_head ** 0.5)
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            attention_probs = F.softmax(attention_scores, dim=-1)
            ref_attention_output = torch.matmul(attention_probs, value)
        
        # Reshape back
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
        
        # Combine original and reference outputs with the learnable scale
        # Note: original_output already includes the residual connection in diffusers
        # so we don't need to add residual again
        combined_output = original_output + self.ref_scale * ref_hidden_states
        
        return combined_output
    
    def _adapt_reference_features(self, reference_states, target_dim):
        """
        Adapt reference features to the expected dimension of the attention layer.
        
        Args:
            reference_states: The features from the image encoder
            target_dim: The expected channel dimension
            
        Returns:
            Adapted reference features with the correct dimensions
        """
        # Get the shape of reference states
        if reference_states.ndim == 4:
            batch_size, channels, height, width = reference_states.shape
            # Flatten spatial dimensions
            reference_states = reference_states.view(batch_size, channels, -1).transpose(1, 2)
        
        # Check if we need to adapt the feature dimension
        feature_dim = reference_states.shape[-1]
        if feature_dim != target_dim:
            # Create feature adapter if it doesn't exist or has wrong dimensions
            if self.feature_adapter is None or self.feature_adapter.in_features != feature_dim:
                # Create a new adapter
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
            
            logger.debug(f"Adapted reference feature from dim={feature_dim} to dim={target_dim}")
            
        return reference_states
    
    def load_original_weights(self, attn_module):
        """
        Initialize the image cross-attention weights from the original attention module.
        
        Args:
            attn_module: Original attention module to copy weights from
        """
        # We will NOT directly copy key and value weights since they might have different dimensions
        # We'll only copy the query and output projections which should have matching dimensions
        with torch.no_grad():
            # Copy query weights (these should always match in dimension)
            self.to_q_ref.weight.copy_(attn_module.to_q.weight)
            
            # For output projection (should be safe to copy)
            self.to_out_ref[0].weight.copy_(attn_module.to_out[0].weight)
            self.to_out_ref[0].bias.copy_(attn_module.to_out[0].bias)
            
            # Initialize with xavier uniform for the key/value projections
            nn.init.xavier_uniform_(self.to_k_ref.weight)
            nn.init.xavier_uniform_(self.to_v_ref.weight)

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
        def default_processor(attn, *args, **kwargs):
            return attn(*args, **kwargs)
        processor.original_processor = default_processor
    
    # Initialize the weights from the original attention module
    processor.load_original_weights(attn_module)
    
    logger.info(f"Created processor for {name} with query_dim={query_dim}, heads={heads}")
    
    return processor