# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mandatory Behavioral Rules

**NEVER GUESS. ALWAYS VERIFY.**

- When something fails, read the error message and investigate the actual cause. Do not speculate.
- NEVER say "can't be done", "not available", "not reachable", or "not found" without first running the command and seeing the failure yourself.
- When a tool/package/URL seems unavailable, try alternative indexes (nightly, --pre), proxy settings (--trusted-host), and different paths before concluding.
- When debugging integration issues, READ THE SOURCE CODE FIRST. Narrow the fault boundary with evidence — do not shotgun-guess at random causes.
- When unsure, say "let me check" and run the command. Do not confabulate.
- When diagnosing a pipeline failure, map the full pipeline, prove each stage healthy or broken, then focus only on the broken stage. Stop testing stages already proven healthy.
- Corporate proxy environments require --trusted-host flags for pip, proxy env vars for git/curl. This is standard — never treat proxy SSL errors as blockers.
- When a build/tool fails, INVESTIGATE the root cause using git log, git blame, PR history, and commit messages BEFORE proposing workarounds. Trace when and why a dependency was introduced. The code and git history are in your hands — use them.
- Never propose "switch to older version" or "cherry-pick to another branch" as a first response to a build failure. First understand WHY it fails and whether it can be fixed in place.

## Project Overview

**torch_npu** (Ascend Extension for PyTorch) adapts Huawei Ascend NPU hardware to PyTorch via the PrivateUse1 backend mechanism. Users do `import torch_npu` and then use `device="npu"` transparently with standard PyTorch APIs.

- Version: 2.12.0 (tracks PyTorch 2.12)
- Runtime dependency: CANN (Compute Architecture for Neural Networks) toolkit
- Requires CPU-only PyTorch (no CUDA build)

## Build Commands

```bash
# Full build (generates wheel in dist/)
bash ci/build.sh --python=3.10

# Build options
bash ci/build.sh --python=3.10 --disable_torchair  # skip torchair compilation
bash ci/build.sh --python=3.10 --disable_rpc        # skip RPC framework
bash ci/build.sh --python=3.10 --enable_lto          # link-time optimization

# Install from built wheel
pip install dist/torch_npu*.whl

# Environment variables for build customization
export _GLIBCXX_USE_CXX11_ABI=1        # CXX11 ABI (default since v2.7)
export DISABLE_INSTALL_TORCHAIR=TRUE    # skip torchair
export DISABLE_RPC_FRAMEWORK=TRUE       # skip RPC
```

Submodules must be initialized first: `git submodule update --init --recursive`

## Testing

Tests are in `test/` and use `torch_npu.testing.testcase.TestCase` as the base class.

```bash
# Install test dependencies
pip install -r test/requirements.txt

# Run a single test file
cd test && python test_autocast.py

# Run a specific test case
cd test && python test_autocast.py -v -k test_autocast_nn_fp32

# Alternative via run_test.py
cd test && python run_test.py -i test_autocast
cd test && python run_test.py -v -i test_autocast -- -k test_autocast_nn_fp32

# Run all non-distributed tests
python ci/access_control_test.py --all

# Run all distributed tests
python ci/access_control_test.py --distributed

# Skip known-failing upstream tests
export DISABLED_TESTS_FILE=./test/unsupported_test_cases/.pytorch-disabled-tests.json

# Sync upstream PyTorch test files (requires network)
cd test && bash get_synchronized_files.sh
```

## Architecture

### Backend Registration (torch_npu/__init__.py)

On `import torch_npu`, the module:
1. Checks no other accelerator (CUDA) is active
2. Calls `torch.utils.rename_privateuse1_backend("npu")` to claim the device name
3. Registers `torch_npu.npu` as the device module
4. Generates standard tensor/module/storage methods for the "npu" backend
5. Monkey-patches `torch.nn.functional` and `torch.nn` for NPU compatibility
6. Registers distributed backends (HCCL, LCCL)
7. Sets up Dynamo compiler backends (torchair, npugraph_ex)
8. Calls `torch_npu._C._initExtension()` to initialize the C++ layer

### C++ Native Layer (torch_npu/csrc/)

- **core/npu/**: Device fundamentals — NPUCachingAllocator (memory), NPUStream, NPUEvent, NPUGuard, NPUGraph
- **aten/**: ATen operator registration and dispatch to NPU kernels
  - `npu_native_functions.yaml` defines the operator schema (backend=NPU)
  - `ops/op_api/` contains ACLNN-based operator implementations
- **framework/**: OpCommand execution pipeline — builds and dispatches ACL operations
- **core/npu/interface/AclInterface.cpp**: Direct ACL (Ascend Computing Language) C library wrapper
- **distributed/**: ProcessGroupHCCL (multi-NPU collective comms), ProcessGroupLCCL

### Operator Flow

Python call → PyTorch dispatcher → NPU registered kernel (csrc/aten/ops/) → OpCommand framework → ACL interface → Ascend hardware

### Key Subsystems

| Directory | Purpose |
|-----------|---------|
| `torch_npu/_inductor/` | Inductor backend for torch.compile; `lowering_fx.py` maps aten ops to NPU |
| `torch_npu/dynamo/` | TorchDynamo backend registration (torchair, npugraph_ex) |
| `torch_npu/npu/` | Python device API: streams, memory, AMP, format handling, graphs |
| `torch_npu/distributed/` | Distributed training (HCCL, RPC) |
| `torch_npu/profiler/` | MSTX-based performance profiling |
| `third_party/op-plugin/` | Custom operator plugin implementations (git submodule) |
| `torchnpugen/` | Build-time code generation (autograd, registrations) |

### Code Generation

Build-time scripts in `torchnpugen/` generate C++ registration code from YAML schemas. The `generate_code.sh` script runs during the build process.

## Code Style

- **Python**: PEP 8, verified with pylint
- **C++**: Google C++ Style Guide (C++17), verified with CppLint/CppCheck
- **Commit messages**: Type-prefixed (feat, fix, refactor, docs, test, etc.)

## Key Gotchas

- Two accelerators cannot coexist: torch_npu requires CPU-only PyTorch (no CUDA)
- CANN environment must be sourced before running: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- NPU has custom memory formats (5D layouts); format conversion happens via `torch_npu.npu.npu_format_cast`
- The `third_party/op-plugin` submodule contains most operator implementations — changes to ops often go there
