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
        scale: float = 0.3,
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
        ref_scale: float = 0.3,
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
        
        reference_states = ref_hidden_states[self.name]
        
        # normalize reference features
        with torch.no_grad():
            ref_mean = reference_states.mean().item()
            ref_std = reference_states.std().item()

            reference_states = (reference_states - reference_states.mean(dim=(0, 1), keepdim=True) * 0.5)
            ref_std_tensor = torch.clamp(reference_states.std(dim=(0, 1), keepdim=True), min=1e-5)
            reference_states = reference_states / ref_std_tensor * 0.8
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # adapt dimensions of reference features
        reference_states = self._adapt_reference_features(reference_states, self.query_dim)
        
        # query from hidden states
        query = self.to_q_ref(hidden_states)
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # key and value from reference features
        key = self.to_k_ref(reference_states)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = self.to_v_ref(reference_states)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # compute cross-attention
        ref_attention_output = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=None,
            dropout_p=0.0, 
            is_causal=False
        )

        ref_attention_output = ref_attention_output.transpose(1, 2).reshape(
            batch_size, -1, self.heads * self.dim_head
        )
        
        # output projection
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
                ref_hidden_states = ref_hidden_states / max(ref_std, 1.0)

        # clamp ref_scale to reasonable values
        safe_ref_scale = min(max(ref_scale, 0.0), 0.5)
        combined_output = original_output + safe_ref_scale * ref_hidden_states
        
        return combined_output
    

    def _adapt_reference_features(self, reference_states, target_dim):
        
        if reference_states.ndim == 4:
            batch_size, channels, height, width = reference_states.shape
            reference_states = reference_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        return reference_states
    

    def load_original_weights(self, attn_module):
        # TODO: Simplify this

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

    query_dim = attn_module.to_q.in_features
    heads = attn_module.heads
    dim_head = attn_module.to_q.out_features // heads
    
    processor = ImageCrossAttentionProcessor(
        name=name,
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head
    )
    
    processor.original_processor = attn_module.processor
    
    processor.load_original_weights(attn_module)
    
    return processor