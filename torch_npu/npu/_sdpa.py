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

    # Early return for zero-length sequences
    if seq_len_q == 0 or seq_len_k == 0 or head_dim == 0:
        empty_tensor = torch.empty(0, device=query.device, dtype=torch.int32)
        return (
            torch.empty_like(query),
            torch.empty(0, device=query.device, dtype=torch.float32),
            empty_tensor, empty_tensor, seq_len_q, seq_len_k,
            torch.tensor(0, dtype=torch.long, device=query.device),
            torch.tensor(0, dtype=torch.long, device=query.device),
            torch.empty(0, device=query.device, dtype=query.dtype),
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    keep_prob = 1.0 - dropout_p

    # Build attention mask for causal mode
    # Note: when is_causal=True, attn_bias is ignored (matches PyTorch behavior)
    atten_mask = None
    sparse_mode = 0
    if is_causal:
        # 3 = down-right aligned causal mask (standard autoregressive)
        sparse_mode = 3
        max_len = max(seq_len_q, seq_len_k)
        atten_mask = torch.triu(
            torch.ones(max_len, max_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
    elif attn_bias is not None:
        if attn_bias.dtype == torch.bool:
            atten_mask = attn_bias
        # else: additive bias handled via pse below

    # Additive attention bias passed as pse (position-specific embedding)
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

    # Build logsumexp from softmax_max for backward compatibility
    if softmax_max.numel() > 0 and softmax_max.dim() == 4:
        logsumexp = softmax_max[..., 0]
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

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
    NPU backward for SDPA.

    npu_fusion_attention is autograd-aware, so its forward pass records the
    backward graph automatically. This explicit backward is only reached if
    the autograd graph was somehow detached.

    We recompute attention via the math path (matmul + softmax) to get gradients.
    This is slower than a fused backward but correct, and the forward path
    (where perf matters most) still uses the NPU-accelerated kernel.
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Recompute via math attention: Q @ K^T * scale -> softmax -> @ V
    # Then use autograd to get gradients
    with torch.enable_grad():
        q = query.detach().requires_grad_(grad_input_mask[0])
        k = key.detach().requires_grad_(grad_input_mask[1])
        v = value.detach().requires_grad_(grad_input_mask[2])

        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            seq_len_q = q.shape[2]
            seq_len_k = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=seq_len_k - seq_len_q + 1,
            )
            attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))
        elif attn_bias is not None and attn_bias.numel() > 0:
            if attn_bias.dtype == torch.bool:
                attn_weight = attn_weight.masked_fill(attn_bias, float("-inf"))
            else:
                attn_weight = attn_weight + attn_bias

        attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
        if dropout_p > 0.0:
            attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
        output = torch.matmul(attn_weight, v)

    grads_to_compute = []
    tensors_for_grad = []
    if grad_input_mask[0]:
        tensors_for_grad.append(q)
    if grad_input_mask[1]:
        tensors_for_grad.append(k)
    if grad_input_mask[2]:
        tensors_for_grad.append(v)

    if tensors_for_grad:
        computed_grads = torch.autograd.grad(output, tensors_for_grad, grad_out)
    else:
        computed_grads = ()

    idx = 0
    grad_query = computed_grads[idx] if grad_input_mask[0] else torch.empty(0, device=query.device)
    if grad_input_mask[0]:
        idx += 1
    grad_key = computed_grads[idx] if grad_input_mask[1] else torch.empty(0, device=query.device)
    if grad_input_mask[1]:
        idx += 1
    grad_value = computed_grads[idx] if grad_input_mask[2] else torch.empty(0, device=query.device)
    grad_attn_bias = torch.empty(0, device=query.device)

    return grad_query, grad_key, grad_value, grad_attn_bias


def register_sdpa_for_npu():
    """Register SDPA overrideable implementations for PrivateUse1 (NPU)."""
    torch.library.impl(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "PrivateUse1",
    )(_sdpa_forward_npu)

    torch.library.impl(
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        "PrivateUse1",
    )(_sdpa_backward_npu)
