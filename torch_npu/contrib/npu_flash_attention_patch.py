"""
Patch for HuggingFace transformers' npu_flash_attention module.

Provides the implementation for npu_flash_attn_with_kvcache() which is
currently a NotImplementedError stub in transformers. This bridges to
torch_npu.npu_incre_flash_attention for efficient autoregressive decoding
with KV cache on Ascend NPU.

Usage:
    # Apply patch BEFORE loading any HF model
    from torch_npu.contrib import npu_flash_attention_patch
    npu_flash_attention_patch.apply()
"""

import math
import warnings
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
    window_size=(-1, -1),
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
        k: New key to append to cache at cache_seqlens position
        v: New value to append to cache at cache_seqlens position
        cache_seqlens: Actual sequence lengths per batch element, shape (B,)
        block_table: Block table for paged attention (not supported on NPU)
        softmax_scale: Scaling factor, defaults to 1/sqrt(head_dim)
        causal: Whether to use causal mask
        window_size: Sliding window size (not supported, will warn if used)
        num_key_value_heads: For GQA, number of KV heads (0 = infer from k_cache)
    """
    head_dim = q.shape[-1]
    num_heads = q.shape[2]
    batch_size = q.shape[0]
    seq_len_q = q.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Infer num_key_value_heads from cache shape instead of defaulting to num_heads
    if num_key_value_heads == 0:
        num_key_value_heads = k_cache.shape[2]

    # Write new KV tokens into cache at cache_seqlens positions
    if k is not None and v is not None and cache_seqlens is not None:
        s_new = k.shape[1]
        for i in range(batch_size):
            seq_pos = int(cache_seqlens[i].item())
            k_cache[i, seq_pos:seq_pos + s_new] = k[i, :s_new]
            v_cache[i, seq_pos:seq_pos + s_new] = v[i, :s_new]
        # Update effective sequence lengths to include new tokens
        cache_seqlens = cache_seqlens + s_new

    # Warn if sliding window is requested (not supported by npu_incre_flash_attention)
    if window_size != (-1, -1) and window_size is not None:
        warnings.warn(
            "npu_flash_attn_with_kvcache does not support sliding window attention. "
            "window_size parameter is ignored, full KV cache will be attended to.",
            stacklevel=2,
        )

    seq_len_kv = k_cache.shape[1]

    # Convert from BSND (HF format) to BSH (npu_incre_flash_attention format)
    q_bsh = q.reshape(batch_size, seq_len_q, num_heads * head_dim)
    k_bsh = k_cache.reshape(batch_size, seq_len_kv, num_key_value_heads * head_dim)
    v_bsh = v_cache.reshape(batch_size, seq_len_kv, num_key_value_heads * head_dim)

    # Build causal mask for multi-token query
    atten_mask = None
    if causal and seq_len_q > 1:
        atten_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool),
            diagonal=seq_len_kv - seq_len_q + 1,
        )

    # Prepare actual_seq_lengths
    actual_seq_lengths = None
    if cache_seqlens is not None:
        actual_seq_lengths = cache_seqlens.tolist()

    output = torch_npu.npu_incre_flash_attention(
        q_bsh,
        k_bsh,
        v_bsh,
        num_heads=num_heads,
        input_layout="BSH",
        scale_value=softmax_scale,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        num_key_value_heads=num_key_value_heads,
    )

    # Convert back from BSH to BSND
    output = output.reshape(batch_size, seq_len_q, num_heads, head_dim)
    return output


def apply():
    """
    Patch HuggingFace transformers' npu_flash_attention module to provide
    the KV cache implementation.

    Must be called BEFORE loading any HF model to ensure the patch takes effect
    before HF's lazy import caching captures the function reference.
    """
    try:
        from transformers.integrations import npu_flash_attention
        npu_flash_attention.npu_flash_attn_with_kvcache = npu_flash_attn_with_kvcache
    except ImportError:
        return

    # Also patch the cached global in modeling_flash_attention_utils
    # if it was already loaded (handles race with lazy import caching)
    try:
        from transformers import modeling_flash_attention_utils as mfau
        if hasattr(mfau, '_flash_with_kvcache_fn') and mfau._flash_with_kvcache_fn is not None:
            mfau._flash_with_kvcache_fn = npu_flash_attn_with_kvcache
    except (ImportError, AttributeError):
        pass
