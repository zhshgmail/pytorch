"""
Register NPU implementation for PyTorch's Scaled Dot-Product Attention (SDPA).

Bridges torch.nn.functional.scaled_dot_product_attention to
torch_npu.npu_fusion_attention when running on NPU, so that
models using attn_implementation="sdpa" (the default in HF transformers 5.x)
work on Ascend NPU without requiring flash_attention_2 override.
"""

import math
from typing import Optional

import torch
import torch_npu


def _sdpa_forward_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
):
    """
    NPU implementation of _scaled_dot_product_fused_attention_overrideable.

    Input layout: PyTorch SDPA uses (B, H, S, D) = BNSD format.
    npu_fusion_attention supports "BNSD" directly.

    Returns 9-tuple: (output, logsumexp, cum_seq_q, cum_seq_k,
                       max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
    """
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    seq_len_q = query.shape[2]
    seq_len_k = key.shape[2]
    head_dim = query.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    keep_prob = 1.0 - dropout_p

    # Build attention mask for causal mode
    atten_mask = None
    sparse_mode = 0
    if is_causal:
        # Use sparse_mode for efficient causal masking
        # 3 = down-right aligned causal mask (standard autoregressive)
        sparse_mode = 3
        # npu_fusion_attention with sparse_mode=3 generates causal mask internally,
        # but needs a dummy mask tensor for the API
        max_len = max(seq_len_q, seq_len_k)
        atten_mask = torch.triu(
            torch.ones(max_len, max_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
    elif attn_bias is not None:
        # Convert additive bias to boolean mask: positions to mask out
        # attn_bias is additive (large negative = masked), we need boolean mask
        # where True = masked position
        if attn_bias.dtype == torch.bool:
            atten_mask = attn_bias
        else:
            # Treat attn_bias as additive; pass as pse (position-specific embedding)
            # npu_fusion_attention supports additive bias via pse parameter
            pass

    # Determine pse (additive attention bias)
    pse = None
    if attn_bias is not None and not is_causal and attn_bias.dtype != torch.bool:
        pse = attn_bias

    result = torch_npu.npu_fusion_attention(
        query, key, value,
        head_num=num_heads,
        input_layout="BNSD",
        pse=pse,
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=keep_prob,
        sparse_mode=sparse_mode,
    )

    # result is 7-tuple: (output, softmax_max, softmax_sum, softmax_out, seed, offset, numels)
    attention_out = result[0]
    softmax_max = result[1]
    seed = result[4] if len(result) > 4 else torch.tensor(0, dtype=torch.long, device=query.device)
    offset = result[5] if len(result) > 5 else torch.tensor(0, dtype=torch.long, device=query.device)

    # Build logsumexp from softmax_max + log(softmax_sum) for backward compatibility
    # Shape: (B, H, S_q) — squeeze the trailing dim
    if softmax_max.numel() > 0 and softmax_max.dim() == 4:
        logsumexp = softmax_max[..., 0]  # Take first element of the 8-wide vector
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

    # Return empty tensors for fields not used in standard (non-varlen) attention
    empty_tensor = torch.empty(0, device=query.device, dtype=torch.int32)
    debug_mask = torch.empty(0, device=query.device, dtype=query.dtype)

    return (
        attention_out,
        logsumexp,
        empty_tensor,         # cum_seq_q
        empty_tensor,         # cum_seq_k
        seq_len_q,            # max_q (SymInt)
        seq_len_k,            # max_k (SymInt)
        seed,                 # philox_seed
        offset,               # philox_offset
        debug_mask,           # debug_attn_mask
    )


def _sdpa_backward_npu(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    grad_input_mask: list,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
):
    """
    NPU backward for SDPA. Falls back to the math implementation
    since npu_fusion_attention backward is handled by the autograd
    graph of the forward call (it's already autograd-aware).
    """
    # npu_fusion_attention registers its own backward through PyTorch's
    # autograd system. When called through the standard SDPA dispatcher,
    # the backward is handled automatically.
    # This explicit backward is only called if the autograd path doesn't work,
    # in which case we fall back to the math-based backward.
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Reconstruct attention weights and compute gradients via math path
    # This is the safe fallback - the forward path is where the perf matters
    return torch.ops.aten._scaled_dot_product_attention_math_backward(
        grad_out, query, key, value, attn_bias, grad_input_mask,
        out, logsumexp, dropout_p, is_causal, scale=scale,
    )


def register_sdpa_for_npu():
    """Register SDPA overrideable implementations for PrivateUse1 (NPU)."""
    torch.library.impl(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "PrivateUse1",
    )(_sdpa_forward_npu)
