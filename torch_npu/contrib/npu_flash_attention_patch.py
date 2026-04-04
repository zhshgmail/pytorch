"""
Patch for HuggingFace transformers' npu_flash_attention module.

Provides the implementation for npu_flash_attn_with_kvcache() which is
currently a NotImplementedError stub in transformers. This bridges to
torch_npu.npu_incre_flash_attention for efficient autoregressive decoding
with KV cache on Ascend NPU.

Usage:
    # Apply patch before using HF inference with KV cache
    from torch_npu.contrib import npu_flash_attention_patch
    npu_flash_attention_patch.apply()
"""

import math
from typing import Optional

import torch
import torch_npu


def npu_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_key_value_heads: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Flash attention with KV cache for autoregressive decoding on NPU.

    Uses torch_npu.npu_incre_flash_attention which is optimized for
    single-token query with full KV cache (the typical decode step).

    Args:
        q: Query tensor, shape (B, S_q, H, D) where S_q is typically 1 for decode
        k_cache: Key cache, shape (B, S_kv, H_kv, D)
        v_cache: Value cache, shape (B, S_kv, H_kv, D)
        k: New key to append to cache (unused, cache is pre-updated)
        v: New value to append to cache (unused, cache is pre-updated)
        cache_seqlens: Actual sequence lengths per batch element
        block_table: Block table for paged attention (not supported yet)
        softmax_scale: Scaling factor, defaults to 1/sqrt(head_dim)
        causal: Whether to use causal mask
        num_key_value_heads: For GQA, number of KV heads (0 = MHA)
    """
    head_dim = q.shape[-1]
    num_heads = q.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if num_key_value_heads == 0:
        num_key_value_heads = num_heads

    # Convert from BSND (HF format) to BSH (npu_incre_flash_attention format)
    # q: (B, S_q, H, D) -> (B, S_q, H*D)
    batch_size = q.shape[0]
    seq_len_q = q.shape[1]
    seq_len_kv = k_cache.shape[1]

    q_bsh = q.reshape(batch_size, seq_len_q, num_heads * head_dim)
    k_bsh = k_cache.reshape(batch_size, seq_len_kv, num_key_value_heads * head_dim)
    v_bsh = v_cache.reshape(batch_size, seq_len_kv, num_key_value_heads * head_dim)

    # Build attention mask for causal mode
    atten_mask = None
    if causal and seq_len_q > 1:
        # For multi-token query with causal mask
        atten_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool),
            diagonal=seq_len_kv - seq_len_q + 1,
        )

    output = torch_npu.npu_incre_flash_attention(
        q_bsh,
        k_bsh,
        v_bsh,
        num_heads=num_heads,
        input_layout="BSH",
        scale_value=softmax_scale,
        atten_mask=atten_mask,
        actual_seq_lengths=cache_seqlens.tolist() if cache_seqlens is not None else None,
        num_key_value_heads=num_key_value_heads,
    )

    # Convert back from BSH to BSND
    output = output.reshape(batch_size, seq_len_q, num_heads, head_dim)
    return output


def apply():
    """
    Patch HuggingFace transformers' npu_flash_attention module to provide
    the KV cache implementation.
    """
    try:
        from transformers.integrations import npu_flash_attention
        npu_flash_attention.npu_flash_attn_with_kvcache = npu_flash_attn_with_kvcache
    except ImportError:
        pass
