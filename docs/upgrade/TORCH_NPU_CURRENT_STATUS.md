# Torch NPU 当前状态总结

> 本文档总结 torch_npu 2.12.0 当前对 PyTorch 和 HuggingFace Transformers 的支持情况。
> 生成日期：2026-04-04

## 1. 算子注册总览

### 1.1 ATen 直接注册算子 (npu_native_functions.yaml)

**共 66 个算子**，分为三类：

#### 标准 PyTorch 算子 (49 个)
| 类别 | 算子 |
|------|------|
| 内存/张量操作 | `clone`, `copy_`, `contiguous`, `empty.*`, `full.*`, `resize_`, `set_.*`, `as_strided`, `_to_copy`, `_lazy_clone` |
| 形状操作 | `squeeze`, `unsqueeze`, `unfold`, `view`, `_reshape_alias` |
| 窗口函数 | `bartlett_window`, `blackman_window`, `hamming_window`, `hann_window` (各含 periodic 变体) |
| 工具函数 | `isnan`, `is_pinned`, `is_set_to`, `tril_indices`, `triu_indices`, `record_stream`, `_pin_memory` |
| 内存管理 | `flatten_dense_tensors`, `_copy_from_and_resize`, `new_empty_strided` |

#### NPU 专有算子 (15 个)
- `npu_format_cast` (6 个变体) — NPU 5D 格式转换
- `npu_change_data_ptr` — 数据指针操作
- `get_npu_format` / `get_storage_size` — 格式/存储查询
- `empty_with_format` / `unsafe_empty_with_format` / `empty_with_swapped_memory` — 带格式的张量创建
- `copy_memory_` — 显式内存拷贝

### 1.2 Inductor Lowering (lowering_fx.py)

**约 50+ 个算子** 有自定义 lowering 规则：

| 类别 | 算子 |
|------|------|
| 逐元素 | `where`, `isinf`, `isnan`, `ceil`, `floor`, `round`, `trunc`, `rsqrt`, `div`, `mul`, `reciprocal`, `pow`, `neg`, `exp`, `log`, `abs`, `sqrt`, `relu`, `gelu`, `tanh`, `sigmoid`, `sign`, `erf` |
| 形状 | `expand`, `view`, `reshape`, `permute`, `squeeze`, `unsqueeze`, `repeat`, `slice`, `select`, `split`, `unbind` |
| 归约 | `sum`, `mean`, `max`, `min`, `prod`, `any`, `amax`, `amin`, `argmax`, `argmin` |
| 类型转换 | `prims.convert_element_type` |

### 1.3 代码生成算子列表 (lowering_op_list.py)

**71+ 个算子**，包括上述 lowering 外的：
- 算术: `add`, `sub`, `mul`, `div` 等
- 逻辑: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `where` 等
- 激活: `relu`, `gelu`, `sigmoid`, `tanh`
- 创建: `full`, `arange`, `scalar_tensor`
- NPU 专有: `npu_dtype_cast`

### 1.4 分解算子 (Decompositions)

**10+ 个算子** 被分解为更基础的操作：
- `aten.expm1`, `aten.erfc` — 数学函数
- `aten.native_dropout` / `aten.native_dropout_backward` — Dropout
- `aten.convolution_backward` — 卷积梯度
- `aten._softmax_backward_data` — Softmax 梯度
- `aten.gelu` / `aten.gelu_backward` — GELU 及梯度

注释掉但已有实现基础的：`npu_rms_norm`, `npu_swiglu`, `npu_rotary_mul` 及其反向

### 1.5 CPU Fallback 算子

**约 80-100 个算子** 在 Inductor 路径下 fallback 到 CPU 执行。

### 1.6 Op-Plugin 算子 (third_party/op-plugin)

作为 git submodule 提供大量算子实现，是实际支持的算子主体（数百个），覆盖大部分 PyTorch 标准 aten 算子。

## 2. CUDA→NPU 翻译层 (transfer_to_npu)

### 2.1 PyTorch API Patch (共 91 个)

| 类别 | 数量 | 示例 |
|------|------|------|
| torch 函数 (张量创建) | 26 | `empty`, `zeros`, `ones`, `full`, `tensor`, `as_tensor` |
| torch 函数 (随机) | 8 | `rand`, `randn`, `randint`, `normal`, `randperm` |
| torch 函数 (窗口) | 5 | `hann_window`, `bartlett_window` 等 |
| torch.cuda API | 22 | `synchronize`, `memory_allocated`, `current_stream`, `set_device` |
| Tensor 方法 | 8 | `to`, `new_empty`, `new_zeros`, `pin_memory` |
| Module 方法 | 2 | `to`, `to_empty` |
| torch.distributed | 7 | `init_process_group` (NCCL→HCCL), `is_nccl_available` |
| torch.fft | 2 | `fftfreq`, `rfftfreq` |
| 特殊替换 | 11 | `CUDAGraph→NPUGraph`, `Tensor.cuda→npu`, `Generator` proxy |

### 2.2 第三方库 Patch (apis_config.json，共 20 个)

| 库 | 最低版本 | Patch 数 | 主要 Patch |
|---|---------|---------|-----------|
| transformers | 4.32.0 | 11 | `TrainingArguments.__post_init__`, `Trainer._load_rng_state/_save_rng_state`, `is_torch_cuda_available`, `is_flash_attn_2_available` 等 |
| trl | 0.7.1 | 2 | `is_bitsandbytes_available`, `get_kbit_device_map` |
| peft | 0.5.0 | 1 | `infer_device` |
| accelerate | 0.22.0 | 6 | `env_command`, `notebook_launcher`, `is_cuda_available`, `PartialState._prepare_backend` 等 |

## 3. NPU 专有贡献模块 (torch_npu/contrib/)

### 3.1 自定义函数 (9 个)
- 检测/几何: `npu_iou/giou/diou/ciou`, `npu_multiclass_nms`, `npu_bbox_coder_*`
- 注意力: `npu_fused_attention`, `npu_fused_attention_with_layernorm`, `fuse_add_softmax_dropout`
- 工具: `matmul_transpose`, `dropout_with_byte_mask`, `roll`

### 3.2 自定义模块 (21 个)
- 注意力: `MultiheadAttention`
- 归一化: `FastBatchNorm1d/2d/3d`, `FastSyncBatchNorm`
- 卷积: `ModulatedDeformConv`, `DCNv2`, `QuantConv2d`
- Dropout: `DropoutWithByteMask`, `NpuFairseqDropout`, `NpuCachedDropout`, `NpuDropPath`
- 量化: `LinearA8W8Quant`, `LinearQuant`, `LinearWeightQuant`
- 其他: `Mish`, `SiLU`, `Focus`, `BiLSTM`, `Prefetcher`

## 4. 已知不支持的测试

`test/unsupported_test_cases/.pytorch-disabled-tests.json` 中有 **31,617 个禁用测试**：

| 类别 | 数量 | 占比 |
|------|------|------|
| UnaryUfuncs | 11,304 | 36% |
| BinaryUfuncs | 4,339 | 14% |
| BwdGradients | 2,004 | 6% |
| FwdGradients | 1,821 | 6% |
| Reductions | 1,775 | 6% |
| Module | 1,246 | 4% |
| SparseAny | 1,159 | 4% |
| ONNX | 1,926 | 6% |
| 其他 | 6,043 | 18% |

大量 Ufunc 测试被禁用，但多数可能是验证/精度问题而非功能缺失。
