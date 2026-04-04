# CANN 算子库盘点

> 基于 ~/workspace/cann/ 下各仓库的最新代码（2026-04-04 git pull）

## 总览

| 仓库 | 算子数 | 核心领域 |
|------|--------|---------|
| **ops-transformer** | 134 | 注意力、MoE、RoPE、多卡通信融合 |
| **ops-nn** | 407 | 激活(64)、归一化(59)、矩阵乘(30)、损失(32)、索引(61) |
| **ops-math** | 270 | 数学运算(176)、类型转换(73)、随机(21) |
| **ops-cv** | 54 | 图像处理(41)、目标检测(13) |
| **ascend-transformer-boost (ATB)** | 67+ | 高性能融合算子、注意力优化 |
| **opbase** | - | 基础框架、调度器 |
| **合计** | **932+** | |

## 1. ops-transformer (134 算子)

### 1.1 注意力算子 (46 个)

**Flash Attention 系列：**
- `flash_attention_score` / `flash_attention_score_grad`
- `incre_flash_attention` (增量/推理用)
- `prompt_flash_attention`
- `fused_infer_attention_score`
- `fused_floyd_attention` / `fused_floyd_attention_grad`

**稀疏/量化注意力：**
- `sparse_flash_attention` / `sparse_flash_attention_grad`
- `kv_quant_sparse_flash_attention`
- `block_sparse_attention` / `block_sparse_attention_grad`
- `swin_attention_score_quant`

**Lightning Attention：**
- `lightning_indexer` / `lightning_indexer_grad`
- `quant_lightning_indexer`
- `dense_lightning_indexer_softmax_lse`

**MLA (Multi-Latent Attention)：**
- `mla_preprocess` / `mla_preprocess_v2`
- `mla_prolog` / `mla_prolog_v2` / `mla_prolog_v3`

**NSA (Native Sparse Attention)：**
- `nsa_compress` / `nsa_compress_attention` / `nsa_compress_grad`
- `nsa_selected_attention` / `nsa_selected_attention_grad`
- `nsa_compress_with_cache` / `nsa_compress_attention_infer` / `nsa_selected_attention_infer`

**Ring Attention：**
- `ring_attention_update`

**KV Cache：**
- `gather_pa_kv_cache`, `scatter_pa_cache`, `scatter_pa_kv_cache`

**其他：**
- `attention_update`, `attention_worker_scheduler`, `attention_worker_combine`
- `chunk_gated_delta_rule`, `recurrent_gated_delta_rule`
- `rain_fusion_attention`

### 1.2 MoE 算子 (28 个)

**路由/初始化：**
- `moe_init_routing` (v1/v2/v3, quant variants)
- `moe_gating_top_k` / `moe_gating_top_k_softmax` (v1/v2)
- `moe_fused_topk`, `moe_re_routing`, `moe_compute_expert_tokens`
- `moe_finalize_routing` (v1/v2, with grad)

**Token 排列：**
- `moe_token_permute` / `moe_token_permute_grad`
- `moe_token_permute_with_ep` / `moe_token_permute_with_routing_map` (及 grad)
- `moe_token_unpermute` / `moe_token_unpermute_grad`
- `moe_token_unpermute_with_ep` / `moe_token_unpermute_with_routing_map` (及 grad)

### 1.3 位置编码算子 (11 个)

- `rotary_position_embedding` / `rotary_position_embedding_grad`
- `apply_rotary_pos_emb`
- `rope_with_sin_cos_cache`
- `rope_quant_kvcache` / `dequant_rope_quant_kvcache`
- `interleave_rope`
- `norm_rope_concat` / `norm_rope_concat_grad`
- `kv_rms_norm_rope_cache` / `qkv_rms_norm_rope_cache`

### 1.4 多卡通信融合 MC2 (36 个)

**MatMul + 通信融合：**
- `all_gather_matmul` (v1/v2)
- `matmul_allto_all`, `matmul_all_reduce`, `matmul_reduce_scatter` (v1/v2)
- `matmul_all_reduce_add_rms_norm`
- `inplace_matmul_all_reduce_add_rms_norm`

**GroupedMatMul + 通信：**
- `grouped_mat_mul_all_reduce`
- `grouped_mat_mul_allto_allv` / `allto_allv_grouped_mat_mul`
- 量化变体: `allto_allv_quant_grouped_mat_mul`, `quant_grouped_mat_mul_allto_allv`

**MoE 分布式：**
- `moe_distribute_dispatch` (v1/v2/v3, setup/teardown)
- `moe_distribute_combine` (v1/v2/v3, setup/teardown)
- `moe_distribute_combine_add_rms_norm`
- `moe_update_expert`

### 1.5 FFN 算子 (5 个)
- `ffn`, `ffn_worker_batching`, `ffn_worker_scheduler`
- `swin_attention_ffn`, `swin_transformer_ln_qkv`

### 1.6 Grouped MatMul (8 个)
- `grouped_matmul`, `grouped_matmul_add`
- `grouped_matmul_swiglu_quant` (v1/v2)
- `quant_grouped_matmul_dequant`, `quant_grouped_matmul_inplace_add`

## 2. ops-nn (407 算子)

### 2.1 激活函数 (64 个)
`relu`, `gelu` (v1/v2), `silu`, `selu`, `elu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax` (v1/v2), `hard_sigmoid`, `hard_swish`, `mish`, `softplus`, `p_relu`, `fast_gelu`, `glu`, `ge_glu_v2`, `swi_glu`, `squared_relu`, `clipped_swiglu`, `fatrelu_mul` 等（含各自 grad）

### 2.2 归一化 (59 个)
`layer_norm` (v1/v3/v4/quant), `batch_norm` (v1/v3), `rms_norm` / `rms_norm_grad` / `rms_norm_quant`, `add_rms_norm` / `add_rms_norm_quant`, `add_layer_norm` / `add_layer_norm_quant`, `group_norm` (v1/v2), `instance_norm`, `ada_layer_norm` (v1/v2/quant), `deep_norm`, `gemma_rms_norm`, `sync_batch_norm` 等

### 2.3 矩阵乘 (30 个)
`mat_mul_v3`, `batch_mat_mul_v3`, `gemm` (v1/v2/v3), `fused_mat_mul`, `quant_matmul`, `quant_batch_matmul` (v3/v4), `weight_quant_batch_matmul` (v1/v2), `sparse_tensor_dense_mat_mul` 等

### 2.4 损失函数 (32 个)
`cross_entropy_loss`, `softmax_cross_entropy_with_logits`, `fused_cross_entropy_loss` 等

### 2.5 优化器 (14 个)
`adam`, `adamw`, `adamax`, `adadelta`, `adagrad`, `rmsprop`, `sgd` 等

### 2.6 卷积 (14 个)、池化 (25 个)、量化 (25 个)、索引 (61 个)、RNN (7 个)

## 3. ops-math (270 算子)

- **数学运算 (176)**: `abs`, `add`, `acos`, `asin`, `atan`, `ceil`, `cos`, `sin`, `exp`, `log`, `sqrt`, `pow`, `div`, `mul`, `cumsum`, `cumprod`, `topk`, `argsort`, `transpose`, `reshape`, `concat`, `split`, `stack`, `pad`, `flip`, `roll` 等
- **类型转换 (73)**: cast, format 转换等
- **随机 (21)**: `dropout` (v1/v3), `uniform`, `normal`, `bernoulli` 等

## 4. ascend-transformer-boost ATB (67+ 算子)

**单算子内核 (31+):** activation, concat, copy, cumsum, elewise, expand, fill, gather, index, matmul, norm, reduce, scatter, slice, softmax, sort, split, transpose 等

**融合内核 (36+):**
- 注意力: `laser_attention`, `unpad_flash_attention`, `pagedattention`, `ring_mla`, `multi_latent_attention`
- FFN: `ffn`, `swi_glu_quant`, `mm_deq_swiglu_quant_mm_deq`
- MoE: `moe_gmm`, `gating`, `fused_add_topk_div`
- 位置编码: `rope`, `rope_grad`, `rms_norm_and_rope_and_reshape_and_cache`
- KV Cache: `kvcache`, `paged_cache_load`, `reshape_and_cache`
- Padding: `pad/unpad_with_hidden_state`

## 5. 关键能力覆盖矩阵

| 能力 | CANN 算子支持 | 覆盖程度 |
|------|-------------|---------|
| Flash Attention | flash_attention_score, incre_flash_attention, prompt_flash_attention | 完整 |
| MLA (DeepSeek v2/v3) | mla_preprocess v1/v2, mla_prolog v1/v2/v3 | 完整 |
| NSA (Native Sparse Attention) | 8 个 NSA 算子 | 完整 |
| MoE 路由/分发 | 28 个 MoE 算子 | 完整 |
| RoPE 位置编码 | 11 个 RoPE 变体 | 完整 |
| KV Cache 管理 | gather/scatter_pa_kv_cache, paged_cache_load | 完整 |
| RMSNorm | rms_norm, add_rms_norm, gemma_rms_norm | 完整 |
| LayerNorm | layer_norm v1/v3/v4/quant | 完整 |
| SwiGLU/GeGLU | swi_glu, ge_glu_v2, clipped_swiglu | 完整 |
| 量化推理 | quant_matmul, weight_quant_batch_matmul | 基本覆盖 |
| 通信融合 | 36 个 MC2 算子 | 完整 |
| Paged Attention | pagedattention, paged_cache_load | 完整 |
| Flex Attention | - | **缺失** |
| FA3/FA4 (PyTorch原生) | - | **需映射** |
