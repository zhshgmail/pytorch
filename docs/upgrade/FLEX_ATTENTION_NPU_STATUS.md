# Flex Attention on NPU — 当前状态与待研究项

> 日期：2026-04-05
> 环境：PyTorch 2.12.0.dev20260404+cpu, torch_npu 2.12.0+git, CANN 8.3.RC1, Ascend910C

## 已解决的问题

### 1. 设备检查硬编码 (已修复)
- **位置**: `torch/nn/attention/flex_attention.py:1474`
- **问题**: `supported_devices = {"cuda", "cpu", "xpu", "hpu"}` 不含 "npu"
- **修复**: monkey-patch `_validate_device` 跳过 NPU 设备

### 2. SizeVarAllocator API 变更 (已修复)
- **位置**: `torch/_inductor/sizevars.py`
- **问题**: `size_hint`, `size_hints`, `symbolic_hint` 在 PyTorch nightly PR #175365 中删除
- **修复**: compat shim 映射到 `optimization_hint` / `optimization_hints`

## 工作的场景

```python
# 无 block_mask 时，flex_attention 在 NPU 上可以工作
from torch.nn.attention.flex_attention import flex_attention
q = torch.randn(1, 8, 16, 64, device="npu:0", dtype=torch.float16)
k = torch.randn(1, 8, 16, 64, device="npu:0", dtype=torch.float16)
v = torch.randn(1, 8, 16, 64, device="npu:0", dtype=torch.float16)
out = flex_attention(q, k, v)  # PASS
```

## 未解决的问题

### 3. block_mask + aclnnSort 崩溃 (待研究)
- **场景**: `flex_attention(q, k, v, block_mask=block_mask)` 
- **create_block_mask 本身**: OK (PASS)
- **使用 block_mask 做 attention**: FAIL

**错误信息**:
```
[E405 05:44:34.642352590 OpParamMaker.cpp:454] operator():
third_party/op-plugin/op_plugin/ops/opapi/SortKernelNpuOpApi.cpp:84
NPU function error: call aclnnSort failed, error code is 561000
EZ9999: Inner Error!
Dst tensor size:4 is less than src tensor size: 8
```

**分析**:
- 错误码 561000 = ACLNN 内部错误
- `Dst tensor size:4 < src tensor size:8` 表明 Sort kernel 的输出 buffer (4 bytes = int32) 小于输入 (8 bytes = int64)
- 发生在 NPU inductor 编译后的 kernel 执行阶段
- `create_block_mask` 内部调用 `torch.argsort(dense_mask_int32, ...)` 成功
- 但编译后的 flex_attention kernel 再次调用 Sort 时失败

**待研究方向**:
1. CANN `aclnnSort` 是否有 int64 input → int32 output 的 bug？
   - 源码位置: `~/workspace/cann/ops-nn/` 中的 Sort 实现
   - 需要查看 `aclnnSort` 的 dtype 处理逻辑
2. torch_npu 的 `SortKernelNpuOpApi.cpp:84` 是否在调用前做了正确的 dtype 检查？
   - 源码: `third_party/op-plugin/op_plugin/ops/opapi/SortKernelNpuOpApi.cpp`
3. NPU inductor codegen 生成的 kernel 是否传了错误的 output tensor dtype？

### 4. HF 模型走 flex_attention 路径失败 (3 的下游)
- **场景**: `AutoModelForCausalLM.from_pretrained(..., attn_implementation="flex_attention")`
- **原因**: HF `masking_utils.py:715` 总是 `create_block_mask` → 传给 flex_attention
- **会触发问题 3**

**第二个错误** (修复 size_hint 后出现):
```
aclrtEventElapsedTime error, error code is 507000
```
这是 NPU event timing 在 torch.compile 路径下的兼容性问题，可能是 inductor 的 performance measurement 代码。

## 深入调查结果 (2026-04-05)

### 关键发现：eager vs compile 行为不同

| 场景 | 结果 |
|------|------|
| `flex_attention(q,k,v)` 无 block_mask | ✅ PASS (所有 size) |
| `flex_attention(q,k,v, block_mask=bm)` 手动调用 | ✅ PASS (16~1024) |
| HF 模型 forward (走 torch.compile) | ❌ FAIL aclnnSort |

standalone 测试通过是因为走 **eager** 路径。HF 模型用 flex_attention 时走 **torch.compile** 路径（通过 `ALL_ATTENTION_FUNCTIONS` 注册），编译后的 kernel 中的 Sort 调用触发 CANN bug。

### CANN 相关发现

Warning 日志明确说：
```
kernel [ArgSort] can not support dtype int32 or int64 on AiCore,
Now this kernel is running on AiCpu.
```

- ArgSort 对 int32/int64 在 AiCore 上不支持，自动降级到 AiCpu
- 后续的 aclnnSort 在编译 kernel 内部被调用，可能没有正确降级
- 错误 `Dst tensor size:4 < src tensor size:8` = int32 output buffer 接收 int64 数据

### SortKernelNpuOpApi.cpp 分析

Line 84 的调用：
```cpp
EXEC_NPU_CMD(aclnnSort, self, argStable, dim, descending, values, indices);
```

indices 创建为 `at::kLong` (int64)，values 保持 self 的 dtype。
torch.compile 路径可能创建了 int32 的 indices tensor 传给 sort_out，导致 size mismatch。

### 根因推断

**torch_npu 的 inductor codegen (triton.py)** 编译 flex_attention kernel 时，生成的 Sort 调用传了错误的 output tensor dtype (int32 而非 int64 indices)。这是 **torch_npu inductor codegen bug**，不是 CANN 算子 bug。

CANN 的 aclnnSort 本身支持 int64 indices（eager 模式验证通过）。问题出在编译后的 kernel 预分配了 int32 buffer。

## 后续行动

1. [ ] 在 `torch_npu/_inductor/codegen/triton.py` 中找到 Sort 相关的 codegen 逻辑
2. [ ] 确认编译后 kernel 中 indices tensor 的 dtype 是否正确
3. [ ] 修复 codegen 中 Sort output dtype 的处理
4. [ ] 或者：在 torch_npu 层面对 flex_attention 的 compile 路径做 dtype 修正 patch
