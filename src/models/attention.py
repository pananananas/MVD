from typing import Optional, Dict, Any
import torch.nn.functional as F
from icecream import ic
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
        img_ref_scale: float = 0.3,
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
        
        self.ref_ln = nn.LayerNorm(self.inner_dim)

        self.feature_adapter = None
        
        self.to_out_ref = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Dropout(dropout)
        ])
        
        # self.ref_scale = nn.Parameter(torch.tensor(img_ref_scale))
        self.ref_scale_val = img_ref_scale

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        ref_hidden_states: Optional[Dict[str, torch.FloatTensor]] = None,
        # ref_scale: float = 0.3,
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
        
        if ref_hidden_states is None or self.name not in ref_hidden_states:
            return original_output
            
        reference_states_input = ref_hidden_states[self.name] # Renamed to avoid confusion
        
        # ic(self.name, "reference_states_input (raw from ImageEncoder)", reference_states_input)

        with torch.no_grad():
            reference_states = reference_states_input.clone()
            reference_states = (reference_states - reference_states.mean(dim=(0, 1), keepdim=True))
            ref_std_tensor = torch.clamp(reference_states.std(dim=(0, 1), keepdim=True), min=1e-6)
            reference_states = reference_states / ref_std_tensor * 0.5
        
        # ic(self.name, "reference_states (normalized for K,V)", reference_states)

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
        ref_hidden_states_output = self.to_out_ref[0](ref_attention_output)
        ref_hidden_states_output = self.to_out_ref[1](ref_hidden_states_output)
        
        # Apply LayerNorm to the output of the reference attention path
        ref_hidden_states_output = self.ref_ln(ref_hidden_states_output)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            ref_hidden_states_output = ref_hidden_states_output.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # ic(self.name, "ref_hidden_states_output (attn out after LayerNorm)", ref_hidden_states_output)

        scaled_ref_contribution = self.ref_scale_val * ref_hidden_states_output
        
        # ic(self.name, "scaled_ref_contribution", scaled_ref_contribution)

        combined_output = original_output + scaled_ref_contribution
        
        # ic(self.name, "combined_output", combined_output)
        
        return combined_output
    

    def _adapt_reference_features(self, reference_states, target_dim):
        
        if reference_states.ndim == 4:
            batch_size, channels, height, width = reference_states.shape
            reference_states = reference_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        return reference_states
    

    def load_original_weights(self, attn_module):

        with torch.no_grad():
            self.to_q_ref.weight.copy_(attn_module.to_q.weight)
            
            self.to_out_ref[0].weight.copy_(attn_module.to_out[0].weight)
            self.to_out_ref[0].bias.copy_(attn_module.to_out[0].bias)
            
            orig_k = attn_module.to_k.weight
            orig_v = attn_module.to_v.weight
            
            k_out, k_in = self.to_k_ref.weight.shape
            ok_out, ok_in = orig_k.shape
            
            if k_out == ok_out and k_in == ok_in:
                self.to_k_ref.weight.copy_(orig_k)
            else:
                if k_in >= ok_in:
                    self.to_k_ref.weight[:, :ok_in].copy_(orig_k[:min(k_out, ok_out), :])
                    if k_in > ok_in:
                        self.to_k_ref.weight[:, ok_in:].zero_()
                else:
                    projection = torch.nn.functional.linear(
                        torch.eye(k_in, device=orig_k.device), 
                        orig_k[:min(k_out, ok_out), :k_in]
                    )
                    self.to_k_ref.weight.copy_(projection)
            
            v_out, v_in = self.to_v_ref.weight.shape
            ov_out, ov_in = orig_v.shape
            
            if v_out == ov_out and v_in == ov_in:
                self.to_v_ref.weight.copy_(orig_v)
            else:
                if v_in >= ov_in:
                    self.to_v_ref.weight[:, :ov_in].copy_(orig_v[:min(v_out, ov_out), :])
                    if v_in > ov_in:
                        self.to_v_ref.weight[:, ov_in:].zero_()
                else:
                    projection = torch.nn.functional.linear(
                        torch.eye(v_in, device=orig_v.device), 
                        orig_v[:min(v_out, ov_out), :v_in]
                    )
                    self.to_v_ref.weight.copy_(projection)


def get_attention_processor_for_module(name, attn_module, img_ref_scale=0.3):
    query_dim = attn_module.to_q.in_features
    heads = attn_module.heads
    dim_head = attn_module.to_q.out_features // heads
    
    processor = ImageCrossAttentionProcessor(
        name=name,
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        img_ref_scale=img_ref_scale
    )
    
    processor.original_processor = attn_module.processor
    
    processor.load_original_weights(attn_module)
    
    return processor