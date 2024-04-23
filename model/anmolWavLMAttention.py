from collections import OrderedDict
import loralib as lora
import torch
from torch import nn
import argparse
from transformers import WavLMModel
import transformers.models.wavlm.modeling_wavlm as wavlm


import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

class CustomWavLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
        lora_dim: int = 768,  # changed dimension of the LORA layer
        lora_rank: int =2, #changed rank of the LORA layer
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # anmol added
        # LORA layers
        self.lora_intermediate_k = lora.Linear(embed_dim, lora_dim, r=lora_rank)
        self.lora_intermediate_q = lora.Linear(embed_dim, lora_dim, r=lora_rank)
        self.lora_intermediate_v = lora.Linear(embed_dim, lora_dim, r=lora_rank)

        
        self.lora_output = lora.Linear(lora_dim, embed_dim, r=lora_rank)


        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        index=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _ = hidden_states.size()

        # first pass of attention layer creates position bias
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )

        # Compute relative position bias:
        # 1) get reshape hidden_states
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) project hidden states
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) compute gate for position bias from projected hidden states
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) apply gate to position bias to compute gated position_bias
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))

        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions
        )

        return attn_output, attn_weights, position_bias

    def torch_multi_head_self_attention(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Union[torch.LongTensor, torch.BoolTensor],
        gated_position_bias: torch.FloatTensor,
        output_attentions: bool,
    ) -> (torch.FloatTensor, torch.FloatTensor):
        """simple wrapper around torch's multi_head_attention_forward function"""
        # self-attention assumes q = k = v
        query = key = value = hidden_states.transpose(0, 1)
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

        # disable bias and add_zero_attn
        bias_k = bias_v = None
        add_zero_attn = False

        query_linear = self.q_proj(query)
        key_linear = self.k_proj(key)
        value_linear = self.v_proj(value)

        query_lora = self.lora_intermediate_q(query)
        key_lora = self.lora_intermediate_k(key)
        value_lora = self.lora_intermediate_v(value)

        query = query_linear + query_lora
        key = key_linear + key_lora
        value = value_linear + value_lora

        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k,
            bias_v,
            add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            output_attentions,
            gated_position_bias,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

        # [Seq_Len, Batch Size, ...] -> [Batch Size, Seq_Len, ...]
        attn_output = attn_output.transpose(0, 1)

        if attn_weights is not None:
            # IMPORTANT: Attention weights are averaged weights
            # here which should not be the case. This is an open issue
            # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )

        return attn_output, attn_weights

        # PyTorch 1.3.0 has F.multi_head_attention_forward defined
        # so no problem with backwards compatibility
        #changed attn_output as attn_output_linear and attention_weights as attn_weights_linear
        # attn_output_linear, attn_weights_linear = F.multi_head_attention_forward(
        #     query,
        #     key,
        #     value,
        #     self.embed_dim,
        #     self.num_heads,
        #     torch.empty([0]),
        #     torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #     bias_k,
        #     bias_v,
        #     add_zero_attn,
        #     self.dropout,
        #     self.out_proj.weight,
        #     self.out_proj.bias,
        #     self.training,
        #     key_padding_mask,
        #     output_attentions,
        #     gated_position_bias,
        #     use_separate_proj_weight=True,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        # )
        # #changed
        # # Pass kqv through LORA layers
        # # query_lora = self.lora_intermediate(query)
        # # key_lora = self.lora_intermediate(key)
        # # value_lora = self.lora_intermediate(value)
        # attn_output_lora, attn_weights_lora = F.multi_head_attention_forward(
        #     query,
        #     key,
        #     value,
        #     self.embed_dim,
        #     self.num_heads,
        #     torch.empty([0]),
        #     torch.cat((self.lora_intermediate_q.bias, self.lora_intermediate_k.bias, self.lora_intermediate_v.bias)),
        #     bias_k,
        #     bias_v,
        #     add_zero_attn,
        #     self.dropout,
        #     self.out_proj.weight,
        #     self.out_proj.bias,
        #     self.training,
        #     key_padding_mask,
        #     output_attentions,
        #     gated_position_bias,
        #     use_separate_proj_weight=True,

        #     q_proj_weight=self.lora_intermediate_q.weight,
        #     k_proj_weight=self.lora_intermediate_k.weight,
        #     v_proj_weight=self.lora_intermediate_v.weight,
        # )
        # # Add the outputs of the linear and LORA layers
        # attn_output = attn_output_linear + attn_output_lora

        # if attn_weights_linear is not None and attn_weights_lora is not None:
        #     attn_weights = attn_weights_linear + attn_weights_lora
        # else:
        #     attn_weights = None
        #     # print("One or both of the attention weights are None")


        # # [Seq_Len, Batch Size, ...] -> [Batch Size, Seq_Len, ...]
        # attn_output = attn_output.transpose(0, 1)

        # if attn_weights is not None:
        #     # IMPORTANT: Attention weights are averaged weights
        #     # here which should not be the case. This is an open issue
        #     # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
        #     attn_weights = attn_weights[:, None].broadcast_to(
        #         attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
        #     )

        # return attn_output, attn_weights

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2

        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets
