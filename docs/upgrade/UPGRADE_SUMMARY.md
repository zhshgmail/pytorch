# torch_npu Transformers 5.x Upgrade Summary

## 1. Project Goal

Upgrade torch_npu to support HuggingFace Transformers 5.x on Ascend NPU.

## 2. Environment Requirements

- **PyTorch 2.12.0 nightly** (CPU, aarch64):
  ```bash
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
  ```
- **CANN 9.0.0-beta.2** (required for flex_attention Sort fix)
- **triton-ascend 3.2.0**
- **transformers >= 5.5.0**
- **Container**: privileged mode, all `/dev/davinci*` devices mounted

## 3. Build Fixes

- **gencode.sh version regex**: strip `+git` suffix with `${1%%+*}`
- **PyTorch nightly API compat fixes**:
  - `pointwise_strategy` removed (PR #178975) — try/except fallback
  - `compiled_autograd_enabled` relocated — try/except with new path
  - `_ConfigEntry` needs `name` arg — try/except both signatures
- **SizeVarAllocator.size_hint/symbolic_hint removed** (PR #175365) — compat shim mapping to `optimization_hint`

## 4. Code Changes (torch_npu)

- **apis_config.json**: removed 3 redundant patches (HF 5.x has native NPU support for `_save_rng_state`, `_load_rng_state`, `is_torch_bf16_gpu_available`)
- **torch_npu/npu/_sdpa.py**: NEW - register SDPA overrideable for NPU, bridging `F.scaled_dot_product_attention` to `npu_fusion_attention`
- **torch_npu/contrib/npu_flash_attention_patch.py**: NEW - implement `npu_flash_attn_with_kvcache` mapping to `npu_incre_flash_attention`
- **torch_npu/__init__.py**:
  - SDPA registration (with try/except)
  - FlexAttention `_validate_device` patch (NPU not in upstream `supported_devices`)
  - `SizeVarAllocator` compat shim
- **torch_npu/distributed/tensor/_pointwise_ops.py**: try/except for removed `pointwise_strategy`
- **torch_npu/distributed/fsdp/_add_fsdp_patch.py**: try/except for relocated `compiled_autograd_enabled`
- **torch_npu/utils/_dynamo.py**: `_ConfigEntry` name arg compat
- **SortKernelNpuOpApi.cpp**: bool->int32 cast, int16->int64 indices buffer handling for `sort_out`

## 5. Verified Features (13/13)

All tested on Ascend 910C (A3) with Llama-3.2-3B:

| # | Feature | Status |
|---|---------|--------|
| 1 | FA2 on NPU | Pass |
| 2 | ALL_ATTENTION_FUNCTIONS interface | Pass |
| 3 | V5 GenerationConfig defaults=None | Pass |
| 4 | V5 processing_class (replaces tokenizer) | Pass |
| 5 | Qwen3 model (GQA) | Pass |
| 6 | Qwen3 MoE model | Pass |
| 7 | DeepSeek V2 (MoE+MLA) | Pass |
| 8 | Speculative Decoding | Pass |
| 9 | Gemma3 Text | Pass |
| 10 | Gemma2 | Pass |
| 11 | Flex Attention (requires CANN 9.0+) | Pass |
| 12 | SDPA on NPU | Pass |
| 13 | KV Cache inference | Pass |

Also verified: BERT inference, Llama-3.2-3B inference (SDPA+GQA), Llama-3.2-3B generate with KV cache, Llama-3.2-3B Trainer training (BF16).

## 6. Key Findings

- HF transformers 5.x already has 36 `is_torch_npu_available()` call sites -- much of the NPU support is native.
- `transfer_to_npu` is still needed for `torch.cuda.empty_cache()` hardcoding in `trainer.py` (2 locations).
- FlexAttention requires CANN >= 9.0.0 -- older CANN has `aclnnSort` int16 indices bug.
- SDPA works via both direct registration (`_scaled_dot_product_fused_attention_overrideable`) and HF's `flash_attention_2` path.

## 7. Remaining Work

- Submit PyTorch upstream PR to add "npu" to flex_attention `supported_devices`.
- Investigate inductor Sort int16 indices for CANN < 9.0 compatibility.
- Performance benchmarking vs CUDA baseline.
- Test with larger models (70B+, multi-chip).
