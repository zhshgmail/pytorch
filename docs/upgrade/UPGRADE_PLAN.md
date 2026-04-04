# Torch NPU 升级计划：适配新版 PyTorch & HuggingFace Transformers

> 生成日期：2026-04-04
> 当前版本：torch_npu 2.12.0 (适配 PyTorch 2.12, transformers 4.40.0)
> 目标：升级 torch_npu 以支持 HuggingFace Transformers 5.x

## 1. 版本选型建议

### 1.1 当前状态

| 组件 | 当前适配版本 | 工作区最新版本 |
|------|------------|--------------|
| torch_npu | 2.12.0 | - |
| PyTorch | 2.12.0 | 2.12.0a0 (main) |
| transformers | 4.40.0 | 5.6.0.dev0 (main) |
| accelerate | ≥0.22.0 | 需确认 |
| trl | ≥0.7.1 | 需确认 |
| peft | ≥0.5.0 | 需确认 |

### 1.2 推荐目标版本

| 组件 | 推荐目标版本 | 理由 |
|------|------------|------|
| **PyTorch** | **2.12.0 (保持)** | torch_npu 已适配 2.12，且工作区 PyTorch 也是 2.12.0a0；升 PyTorch 大版本风险高、收益低 |
| **transformers** | **5.6.0 (稳定版发布后)** | 5.x 是大版本升级，移除了 TF/JAX 后端，PyTorch 成唯一后端；且已有原生 `is_torch_npu_available()` 支持 |
| **accelerate** | **≥1.0.0** | transformers 5.x 依赖新版 accelerate |
| **trl** | **≥0.15.0** | 配合 transformers 5.x |
| **peft** | **≥0.14.0** | 配合 transformers 5.x |

### 1.3 为什么不升 PyTorch？

- torch_npu 已经跟踪 PyTorch 2.12，版本已经是最新
- 升级 PyTorch 大版本涉及 C++ ABI、dispatcher 变更、算子 schema 变更，工作量巨大
- transformers 5.x 仅要求 `torch>=2.4`，当前 2.12 完全满足
- **结论：保持 PyTorch 2.12，专注升级 transformers 生态**

## 2. 关键发现：Transformers 5.x 的变化

### 2.1 利好消息：原生 NPU 支持增强

Transformers 5.x 已在多处原生支持 NPU：

```python
# transformers/utils/import_utils.py 已有：
def is_torch_npu_available():  # 原生检测 NPU
def is_torch_bf16_supported()  # 含 NPU 路径

# transformers/testing_utils.py 已有：
@require_torch_npu  # NPU 测试装饰器

# transformers/integrations/npu_flash_attention.py 已有：
npu_flash_attn_func()          # NPU Flash Attention (已实现)
npu_flash_attn_varlen_func()   # 变长 Flash Attention (已实现)
npu_flash_attn_with_kvcache()  # KV Cache 版本 (未实现，NotImplementedError)
```

**这意味着 `transfer_to_npu` 中部分 patch 在新版中可能不再需要。**

### 2.2 V5 Breaking Changes

1. **移除 TF/JAX** — PyTorch 成唯一后端（对 NPU 有利，减少歧义）
2. **Dynamic Weight Loading** — 新 `WeightConverter` API（需测试兼容）
3. **Tokenizer 整合** — 单文件 tokenizer（无 NPU 影响）
4. **模型目录结构变更** — `modeling_*.py` 重组（可能影响 monkey-patch 路径）

### 2.3 仍存在的 CUDA 硬编码问题

```python
# trainer.py 中仍有：
torch.cuda.empty_cache()           # 第 2747, 2757 行
torch.cuda.random.get_rng_state()  # 第 3147-3152 行 (RNG 状态)
torch.cuda.random.set_rng_state()  # 第 3548-3549 行
```

这些需要通过 `transfer_to_npu` 或贡献 PR 到上游 transformers 来解决。

## 3. 详细升级计划

### Phase 0: 前置准备 (1 周)

#### 0.1 环境搭建
- [ ] 创建升级开发环境（独立 venv/conda）
- [ ] 安装 PyTorch 2.12 CPU-only
- [ ] 安装 transformers 5.6.0.dev0（从源码）
- [ ] 安装对应版本的 accelerate, trl, peft

#### 0.2 基线测试
- [ ] 运行 torch_npu 现有 transformers 相关测试，记录 baseline
  ```bash
  cd test && python dynamo/test_model_output.py
  cd test && python onnx/dynamo/test_dynamo_with_onnxruntime_backend.py
  ```
- [ ] 用 transformers 4.40.0 跑一个基准模型（如 Llama-2-7B）记录正确性和性能

---

### Phase 1: apis_config.json 更新 (1-2 周)

这是升级的核心工作——保持翻译层与新版本 API 同步。

#### 1.1 审计 transformers 5.x API 路径变更

需要逐一检查 `apis_config.json` 中的每个 patch 目标：

| 当前 Patch | 检查项 | 操作 |
|-----------|--------|------|
| `transformers.training_args.TrainingArguments.__post_init__` | 类是否还在？方法签名变了吗？ | 验证并更新 |
| `transformers.trainer.Trainer._load_rng_state` | 方法是否重构？ | 验证并更新 |
| `transformers.trainer.Trainer._save_rng_state` | 方法是否重构？ | 验证并更新 |
| `transformers.trainer_utils.TrainerMemoryTracker.start` | 类是否还在？ | 验证并更新 |
| `transformers.utils.import_utils.is_torch_cuda_available` | **可能不需要了**——5.x 已有 `is_torch_npu_available` | 评估是否可移除 |
| `transformers.utils.import_utils.is_mamba_ssm_available` | 路径是否变了？ | 验证 |
| `transformers.utils.import_utils.is_causal_conv1d_available` | 路径是否变了？ | 验证 |
| `transformers.utils.import_utils.is_torch_bf16_gpu_available` | **可能不需要了**——5.x 有 NPU BF16 检测 | 评估 |
| `transformers.utils.import_utils.is_torch_tf32_available` | 路径是否变了？ | 验证 |
| `transformers.utils.import_utils.is_bitsandbytes_available` | 路径是否变了？ | 验证 |
| `transformers.utils.import_utils.is_flash_attn_2_available` | 5.x 新增了 FA3/FA4 检测 | 验证并可能新增 FA3/FA4 patch |

```bash
# 检查命令示例
cd ~/workspace/torch/transformers
grep -rn "is_torch_cuda_available" src/transformers/utils/import_utils.py
grep -rn "is_torch_npu_available" src/transformers/utils/import_utils.py
grep -rn "class TrainingArguments" src/transformers/training_args.py
```

#### 1.2 检查新增的 CUDA 硬编码

```bash
# 在 transformers 5.x 中搜索 CUDA 硬编码
cd ~/workspace/torch/transformers
grep -rn "torch\.cuda\." src/transformers/trainer.py
grep -rn "torch\.cuda\." src/transformers/training_args.py
grep -rn '"cuda"' src/transformers/trainer.py
grep -rn "\.cuda()" src/transformers/ --include="*.py" | grep -v test | grep -v __pycache__
```

对发现的新硬编码，评估是否需要添加新 patch。

#### 1.3 更新 accelerate/trl/peft patch

同样审计这三个库的 API 路径变更：

```bash
# accelerate
pip show accelerate  # 确认版本
grep -rn "is_cuda_available" <accelerate_path>/utils/imports.py

# trl
grep -rn "is_bitsandbytes_available" <trl_path>/import_utils.py

# peft
grep -rn "infer_device" <peft_path>/utils/other.py
```

#### 1.4 更新版本门控

```json
// apis_config.json 中 version 字段需要更新
"transformers": {
  "version": "5.0.0",  // 更新最低版本
  ...
}
```

考虑是否需要同时支持 4.x 和 5.x，如果需要，可能要实现版本分支逻辑。

---

### Phase 2: 算子补全 (2-4 周)

#### 2.1 确认 transformers 5.x 使用的新算子

```bash
# 搜索 transformers 中使用的 torch 算子
cd ~/workspace/torch/transformers
grep -rn "torch\.ops\." src/transformers/ --include="*.py" | sort -u
grep -rn "F\.\|nn\.functional\." src/transformers/ --include="*.py" | grep -v test | sort -u | head -50
```

重点关注：
- **Scaled Dot-Product Attention (SDPA)** — `torch.nn.functional.scaled_dot_product_attention`
  - 这是 transformers 5.x 的默认注意力实现
  - torch_npu 需确保 SDPA dispatcher 注册到 NPU
  - CANN 有 `flash_attention_score` 可以映射

- **Flash Attention 3/4** — transformers 5.x 支持 FA3/FA4
  - 检查是否有 NPU 路径

- **Flex Attention** — PyTorch 2.12 新增
  - CANN 目前 **没有对应算子**
  - 短期：在 torch_npu 注册 fallback（用 SDPA 组合实现）
  - 中期：提需求给 CANN 团队

#### 2.2 SDPA → NPU 映射 (最高优先级)

transformers 5.x 大量使用 `F.scaled_dot_product_attention`，这是最关键的算子。

**映射方案：**

```
PyTorch SDPA
  ├─ _scaled_dot_product_flash_attention (CUDA)
  │   └→ 映射到 CANN flash_attention_score
  ├─ _scaled_dot_product_efficient_attention (CUDA)
  │   └→ 映射到 CANN flash_attention_score (统一实现)
  ├─ _scaled_dot_product_attention_math (CPU fallback)
  │   └→ 保持 fallback 或用 CANN 算子组合
  └─ _scaled_dot_product_fused_attention_overrideable (自定义后端)
      └→ **这是 NPU 应该注册的入口**
```

**实现步骤：**
1. 在 `torch_npu/csrc/aten/ops/` 注册 `_scaled_dot_product_fused_attention_overrideable`
2. 内部调用 CANN `flash_attention_score` / `incre_flash_attention`
3. 处理 causal mask、GQA、变长序列等变体

#### 2.3 完善 npu_flash_attention 集成

transformers 5.x 已有 `integrations/npu_flash_attention.py`，但 KV Cache 版本未实现：

```python
# 需要实现：
def npu_flash_attn_with_kvcache(
    q, k_cache, v_cache, k=None, v=None,
    cache_seqlens=None, block_table=None, ...
):
    # 映射到 CANN incre_flash_attention 或 gather_pa_kv_cache + flash_attention_score
    ...
```

CANN 已有 `incre_flash_attention`、`gather_pa_kv_cache`、`scatter_pa_kv_cache`，完全可以支持。

#### 2.4 新模型所需算子检查

transformers 5.x 新增的模型中，检查是否有 torch_npu 未覆盖的算子：

| 新模型 | 关键算子需求 | CANN 支持 |
|--------|-------------|----------|
| Qwen3/3.5 | SwiGLU, RoPE, GQA, Flash Attention | 完整支持 |
| Qwen3.5-MoE | MoE routing, token permute | 完整支持 (28 个 MoE 算子) |
| Llama 3.x | RoPE, GQA, SDPA | 完整支持 |
| Mixtral | MoE + Flash Attention | 完整支持 |
| DeepSeek-V2/V3 | MLA (Multi-Latent Attention) | 完整支持 (mla_preprocess/prolog) |
| Gemma 2/3 | RMSNorm, SlidingWindow Attention | 完整支持 (gemma_rms_norm) |
| RWKV-6/7 | 自定义 CUDA kernel (wkv) | **需要 AscendC 实现** |
| xLSTM | 自定义 recurrence kernel | **需要 AscendC 实现** |

---

### Phase 3: transfer_to_npu 精简与增强 (1-2 周)

#### 3.1 评估哪些 patch 可以移除

transformers 5.x 原生 NPU 支持增强后，部分 patch 可能多余：

| Patch | 5.x 是否还需要 | 原因 |
|-------|---------------|------|
| `is_torch_cuda_available` | **可能不需要** | 5.x 已有 `is_torch_npu_available` |
| `is_torch_bf16_gpu_available` | **可能不需要** | 5.x `is_torch_bf16_supported` 含 NPU |
| `TrainingArguments.__post_init__` | **需要验证** | 内部可能仍硬编码 cuda |
| `Trainer._load_rng_state` | **可能需要** | RNG 仍用 cuda API |
| `Trainer._save_rng_state` | **可能需要** | 同上 |

#### 3.2 新增必要的 patch

```python
# trainer.py 中新增的 CUDA 调用需要 patch：
"transformers.trainer.Trainer.training_step"  # 如果包含新的 cuda 调用
"transformers.generation.utils.GenerationMixin.generate"  # 如有 cuda 硬编码
```

#### 3.3 考虑贡献上游

长期来看，应该向 HuggingFace transformers 提 PR：
- 将 `torch.cuda.empty_cache()` 改为 device-agnostic 写法
- 将 RNG 状态管理改为支持自定义后端
- 完善 `npu_flash_attn_with_kvcache()` 实现

---

### Phase 4: 测试验证 (2-3 周)

#### 4.1 单元测试更新

```bash
# 更新测试依赖
# test/requirements.txt: transformers==5.6.0

# 运行现有测试
cd test && python dynamo/test_model_output.py
cd test && python dynamo/test_repros.py  # HF-specific patterns
cd test && python nn/test_transformer_layers.py
```

#### 4.2 端到端模型验证

优先验证以下模型（按使用频率排序）：

| 优先级 | 模型 | 验证内容 |
|--------|------|---------|
| P0 | Llama-3-8B | 推理 + 训练 (SFT) |
| P0 | Qwen2.5-7B | 推理 + 训练 |
| P1 | Mixtral-8x7B | MoE 推理 + 训练 |
| P1 | Gemma-2-9B | 推理 + 训练 |
| P2 | DeepSeek-V2-Lite | MLA 推理 |
| P2 | Qwen3.5-MoE | MoE 路由完整验证 |
| P3 | Vision models (Qwen-VL, LLaVA) | 多模态推理 |

每个模型验证：
1. `from_pretrained` 加载
2. `model.to("npu")` 或 `transfer_to_npu`
3. 推理正确性（与 CPU/CUDA 对比 logits）
4. 训练一个 epoch（loss 收敛验证）
5. `Trainer` API 训练流程

#### 4.3 性能基准

- 与 transformers 4.40.0 + torch_npu 对比推理延迟
- 与 transformers 4.40.0 + torch_npu 对比训练吞吐
- 确保无性能回退

---

### Phase 5: 文档与发布 (1 周)

- [ ] 更新 CLAUDE.md 中的版本信息
- [ ] 更新 README.md 版本兼容表
- [ ] 编写升级指南（从 transformers 4.x → 5.x）
- [ ] 更新 test/requirements.txt
- [ ] 更新 apis_config.json version 字段

## 4. 风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| transformers 5.x 内部 API 路径大量变更 | apis_config.json 大面积失效 | 5.x 原生 NPU 支持增强，部分 patch 可直接移除 |
| 新模型使用 Flex Attention | 部分模型无法运行 | 短期 fallback 到 SDPA math 实现，中期实现 NPU Flex Attention |
| SDPA dispatcher 未注册到 NPU | 大量模型推理失败 | Phase 2.2 是最高优先级，必须完成 |
| accelerate 大版本升级 | 分布式训练可能 break | 同步更新 accelerate patch |
| RWKV/xLSTM 需要自定义 kernel | 这些模型无法支持 | 明确标注为不支持，后续补充 AscendC kernel |

## 5. 时间线总结

```
Week 1:     Phase 0 — 环境搭建 + 基线测试
Week 2-3:   Phase 1 — apis_config.json 审计更新
Week 3-6:   Phase 2 — SDPA 算子映射 + KV Cache 实现 (与 Phase 1 并行)
Week 5-6:   Phase 3 — transfer_to_npu 精简
Week 7-9:   Phase 4 — 测试验证
Week 10:    Phase 5 — 文档发布
```

**预估总工期：8-10 周** (2-3 人并行可缩短到 5-6 周)

## 6. 附录

### 相关文档
- [当前状态总结](./TORCH_NPU_CURRENT_STATUS.md)
- [CANN 算子库盘点](./CANN_OPERATOR_INVENTORY.md)

### 关键代码位置
| 文件 | 作用 |
|------|------|
| `torch_npu/contrib/apis_config.json` | 第三方库 patch 配置 |
| `torch_npu/contrib/transfer_to_npu.py` | CUDA→NPU 翻译引擎 |
| `torch_npu/csrc/aten/npu_native_functions.yaml` | NPU 算子注册 |
| `torch_npu/_inductor/lowering_fx.py` | Inductor lowering 规则 |
| `third_party/op-plugin/` | 算子插件实现 |
| `test/requirements.txt` | 测试依赖版本 |
| HF `integrations/npu_flash_attention.py` | HF 侧 NPU 集成点 |
| HF `utils/import_utils.py` | HF 侧硬件检测 |
| HF `trainer.py` | CUDA 硬编码需审计 |
