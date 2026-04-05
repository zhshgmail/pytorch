---
name: torch-npu-hf-upgrade
description: Build, deploy, and verify torch_npu for HuggingFace Transformers 5.x on Ascend NPU (A3 910C)
---

# Torch NPU HF Upgrade Skill

Automates the full workflow to build torch_npu from source, deploy to an A3 NPU container, and verify HuggingFace Transformers 5.x compatibility.

## When to Use This Skill

Use this skill when you need to:
- Build torch_npu from the current branch and deploy to A3
- Run the HF transformers 5.x compatibility test suite on NPU
- Verify a specific HF model works on NPU
- Check NPU resource usage before/after tests

## Prerequisites

- A3 container `torch-npu-dev` running (privileged, all 16 NPU chips)
- SSH access to 7.150.11.210 via a3_exec.py
- CANN >= 9.0.0-beta.2 installed in container
- triton-ascend >= 3.2.0 installed in container

## Commands

### Full build + deploy + test

```bash
python skills/torch-npu-hf-upgrade/scripts/build_deploy_test.py
```

This will:
1. Check NPU HBM usage (abort if chip 0 > 50GB used)
2. Push current branch to personal remote
3. Pull on A3 container
4. Build torch_npu wheel
5. Install wheel + copy Python patches
6. Run the full verification suite (13 tests)
7. Report results

### Deploy only (skip build, use existing wheel)

```bash
python skills/torch-npu-hf-upgrade/scripts/build_deploy_test.py --deploy-only
```

### Test only (skip build and deploy)

```bash
python skills/torch-npu-hf-upgrade/scripts/build_deploy_test.py --test-only
```

### Single model test

```bash
python skills/torch-npu-hf-upgrade/scripts/build_deploy_test.py --test-model llama
```

Available models: bert, llama, qwen3, qwen3-moe, deepseek-v2, gemma2, gemma3

## Environment Details

- **Host**: 7.150.11.210
- **Container**: torch-npu-dev (privileged, workdir /data2/z00637938/torch-npu-dev)
- **Source on container**: /data2/z00637938/torch-npu-dev/torch_npu
- **Installed packages**: /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/
- **CANN**: /usr/local/Ascend/cann/
- **Model weights**: /data/nfs/model/

## Safety

- Always checks NPU HBM before running tests
- Uses single chip (npu:0) to avoid interfering with other users
- Cleans up HBM after each test (del model + torch.npu.empty_cache())
- Aborts if chip 0 HBM > 50GB (other workloads running)
