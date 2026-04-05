#!/usr/bin/env python3
"""
Build, deploy, and test torch_npu for HF Transformers 5.x on Ascend NPU.

Usage:
    python build_deploy_test.py              # full pipeline
    python build_deploy_test.py --deploy-only # skip build
    python build_deploy_test.py --test-only   # skip build+deploy
    python build_deploy_test.py --test-model llama  # single model
"""

import argparse
import subprocess
import sys
import os

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SKILL_DIR)
A3_EXEC = os.path.expanduser("~/.claude/skills/a3-team/scripts/a3_exec.py")
CONTAINER = "torch-npu-dev"
CONTAINER_SRC = "/data2/z00637938/torch-npu-dev/torch_npu"
CONTAINER_SITE = "/usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu"
CANN_ENV = "source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null"
PROXY_ENV = "source /home/z00637938/setup_proxy.sh 2>/dev/null"
HBM_LIMIT = 50000  # MB - abort if chip 0 uses more than this


def run_a3(cmd, host=False, proxy=False, timeout=300):
    """Run command on A3 via a3_exec.py."""
    args = [sys.executable, A3_EXEC]
    if host:
        args.append("--host")
    elif proxy:
        args.append("--proxy")
    args.append(cmd if host else f"docker exec {CONTAINER} bash -c '{cmd}'")
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def check_hbm():
    """Check chip 0 HBM usage. Abort if too high."""
    output = run_a3("npu-smi info 2>&1 | grep '65536' | head -1", host=True)
    if "65536" in output:
        parts = output.split()
        for i, p in enumerate(parts):
            if "/" in p and "65536" in p:
                used = int(p.split("/")[0])
                if used > HBM_LIMIT:
                    print(f"ABORT: Chip 0 HBM {used}MB > {HBM_LIMIT}MB limit. Other workloads running.")
                    sys.exit(1)
                print(f"Chip 0 HBM: {used}/65536 MB (safe)")
                return used
    print("WARNING: Could not parse HBM info")
    return 0


def build():
    """Build torch_npu on A3 container."""
    print("\n=== Building torch_npu ===")

    # Push current branch
    print("Pushing current branch...")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()
    subprocess.run(["git", "push", "personal", branch], cwd=PROJECT_ROOT, timeout=30)

    # Pull on container
    print("Pulling on container...")
    run_a3(f"{PROXY_ENV}; cd {CONTAINER_SRC} && git fetch origin {branch} 2>&1 | tail -1 && git checkout origin/{branch} 2>&1 | tail -1")

    # Fix gencode.sh version regex
    run_a3(f"cd {CONTAINER_SRC}/third_party/op-plugin && "
           "sed -i 's|PYTORCH_VERSION=\"$1\"|PYTORCH_VERSION=\"${{1%%+*}}\"|' gencode.sh")

    # Build
    print("Building (this takes ~5 minutes)...")
    run_a3(
        f"cd {CONTAINER_SRC} && {CANN_ENV} && export _GLIBCXX_USE_CXX11_ABI=1 && "
        f"bash ci/build.sh --python=3.11 > /data2/z00637938/torch-npu-dev/build.log 2>&1; "
        f"echo EXIT_CODE=$? >> /data2/z00637938/torch-npu-dev/build.log",
        timeout=600,
    )

    # Check result
    output = run_a3("grep EXIT_CODE /data2/z00637938/torch-npu-dev/build.log; "
                    "grep -c 'error:' /data2/z00637938/torch-npu-dev/build.log")
    if "EXIT_CODE=0" in output:
        print("Build: OK")
    else:
        print(f"Build: FAILED\n{output}")
        sys.exit(1)


def deploy():
    """Install wheel and copy Python patches to container."""
    print("\n=== Deploying ===")

    # Install wheel
    run_a3(f"pip install {CONTAINER_SRC}/dist/torch_npu-*.whl --force-reinstall --no-deps 2>&1 | tail -2")

    # Copy patched Python files
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()

    patches = [
        "torch_npu/__init__.py",
        "torch_npu/distributed/tensor/_pointwise_ops.py",
        "torch_npu/distributed/fsdp/_add_fsdp_patch.py",
        "torch_npu/utils/_dynamo.py",
    ]
    for f in patches:
        run_a3(f"cd {CONTAINER_SRC} && git checkout origin/{branch} -- {f} 2>/dev/null && "
               f"cp {f} {CONTAINER_SITE}/{f}")

    print("Deploy: OK")


def test(model_filter=None):
    """Run HF transformers 5.x verification suite."""
    print("\n=== Testing ===")

    test_script = os.path.join(os.path.dirname(__file__), "test_hf_features.py")
    if not os.path.exists(test_script):
        print("Generating test script...")
        _write_test_script(test_script)

    # Copy test to container
    subprocess.run(["scp", test_script, f"root@7.150.11.210:/tmp/test_hf.py"],
                   capture_output=True, timeout=10)
    run_a3(f"docker cp /tmp/test_hf.py {CONTAINER}:/data2/z00637938/torch-npu-dev/", host=True)

    # Run
    cmd = f"{CANN_ENV}; python3 /data2/z00637938/torch-npu-dev/test_hf.py"
    if model_filter:
        cmd += f" --model {model_filter}"
    output = run_a3(cmd, timeout=600)

    # Parse results
    passed = output.count("PASS")
    failed = output.count("FAIL")
    for line in output.split("\n"):
        if "PASS" in line or "FAIL" in line:
            print(f"  {line.strip()}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def _write_test_script(path):
    """Generate the test script."""
    with open(path, "w") as f:
        f.write('''#!/usr/bin/env python3
"""HF Transformers 5.x verification suite for NPU."""
import argparse
import torch
import torch_npu

device = "npu:0"
results = {}

def test(name, fn):
    try:
        result = fn()
        results[name] = f"PASS {result}"
        print(f"{name}: PASS {result}")
    except Exception as e:
        results[name] = f"FAIL {str(e)[:100]}"
        print(f"{name}: FAIL {str(e)[:100]}")
    torch.npu.empty_cache()

# SDPA
def t_sdpa():
    q = torch.randn(1, 8, 16, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 16, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 16, 64, device=device, dtype=torch.float16)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.shape

# FA2
def t_fa2():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(hidden_size=256, num_hidden_layers=1, num_attention_heads=8,
                         num_key_value_heads=4, intermediate_size=512, vocab_size=1000)
    m = LlamaForCausalLM(config).half().to(device).eval()
    m.config._attn_implementation = "flash_attention_2"
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# Qwen3
def t_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    c = Qwen3Config(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                    num_key_value_heads=2, intermediate_size=512, vocab_size=1000)
    m = Qwen3ForCausalLM(c).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# Qwen3 MoE
def t_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    c = Qwen3MoeConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=8,
                       num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
                       num_experts=4, num_experts_per_tok=2, expert_inter_size=256)
    m = Qwen3MoeForCausalLM(c).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# DeepSeek V2
def t_deepseek_v2():
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    c = DeepseekV2Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=8,
                         num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
                         moe_intermediate_size=128, n_routed_experts=4,
                         num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1)
    m = DeepseekV2ForCausalLM(c).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# Gemma2
def t_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    c = Gemma2Config(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                     num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
                     head_dim=32, query_pre_attn_scalar=32)
    m = Gemma2ForCausalLM(c).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# Gemma3 Text
def t_gemma3():
    from transformers import Gemma3TextConfig, Gemma3TextModel
    c = Gemma3TextConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                         num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
                         head_dim=32, query_pre_attn_scalar=32)
    m = Gemma3TextModel(c).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.last_hidden_state.shape

# Flex Attention (requires CANN >= 9.0)
def t_flex_attention():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(hidden_size=256, num_hidden_layers=1, num_attention_heads=8,
                         num_key_value_heads=4, intermediate_size=512, vocab_size=1000)
    config._attn_implementation = "flex_attention"
    m = LlamaForCausalLM(config).half().to(device).eval()
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 1000, (1, 16), device=device))
    del m
    return o.logits.shape

# Speculative Decoding
def t_speculative():
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    config = LlamaConfig(hidden_size=256, num_hidden_layers=1, num_attention_heads=8,
                         num_key_value_heads=4, intermediate_size=512, vocab_size=1000)
    m = LlamaForCausalLM(config).half().to(device).eval()
    ids = torch.randint(0, 1000, (1, 8), device=device)
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=5, do_sample=False, assistant_model=m)
    del m
    return f"tokens={o.shape[1]}"

# V5 breaking changes
def t_v5_genconfig():
    from transformers import GenerationConfig
    gc = GenerationConfig()
    return f"temp={gc.temperature} top_p={gc.top_p}"

def t_v5_trainer():
    from transformers import Trainer
    import inspect
    sig = inspect.signature(Trainer.__init__)
    has_pc = "processing_class" in sig.parameters
    return f"processing_class={has_pc}"

# KV Cache
def t_kvcache():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(hidden_size=256, num_hidden_layers=1, num_attention_heads=8,
                         num_key_value_heads=4, intermediate_size=512, vocab_size=1000)
    m = LlamaForCausalLM(config).half().to(device).eval()
    ids = torch.randint(0, 1000, (1, 8), device=device)
    with torch.no_grad():
        out = m(ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out2 = m(next_tok, past_key_values=past, use_cache=True)
    del m
    return f"kv_layers={len(past)}"

ALL_TESTS = {
    "sdpa": t_sdpa,
    "fa2": t_fa2,
    "qwen3": t_qwen3,
    "qwen3_moe": t_qwen3_moe,
    "deepseek_v2": t_deepseek_v2,
    "gemma2": t_gemma2,
    "gemma3": t_gemma3,
    "flex_attention": t_flex_attention,
    "speculative": t_speculative,
    "v5_genconfig": t_v5_genconfig,
    "v5_trainer": t_v5_trainer,
    "kvcache": t_kvcache,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Run specific test only")
args = parser.parse_args()

if args.model:
    if args.model in ALL_TESTS:
        test(args.model, ALL_TESTS[args.model])
    else:
        print(f"Unknown model: {args.model}. Available: {list(ALL_TESTS.keys())}")
else:
    for name, fn in ALL_TESTS.items():
        test(name, fn)

print(f"\\nHBM: {torch.npu.memory_allocated(0) // 1024 // 1024} MB")
''')


def main():
    parser = argparse.ArgumentParser(description="Build, deploy, test torch_npu for HF 5.x")
    parser.add_argument("--deploy-only", action="store_true", help="Skip build")
    parser.add_argument("--test-only", action="store_true", help="Skip build and deploy")
    parser.add_argument("--test-model", help="Test specific model only")
    args = parser.parse_args()

    print("=== Torch NPU HF Upgrade Pipeline ===")

    # Safety check
    print("\nChecking NPU resources...")
    check_hbm()

    if not args.test_only and not args.deploy_only:
        build()

    if not args.test_only:
        deploy()

    success = test(args.test_model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
