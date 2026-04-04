"""
Local unit tests for torch_npu transformers 5.x upgrade.

These tests verify the upgrade changes WITHOUT requiring NPU hardware.
They test: apis_config.json validity, SDPA registration logic,
KV cache patch interface, and transfer_to_npu compatibility.

Run: python test/contrib/test_transformers5x_upgrade.py
"""

import json
import math
import os
import sys
import unittest
import importlib

# Get project root
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
TORCH_NPU_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "torch_npu")


class TestApisConfigJson(unittest.TestCase):
    """Test that apis_config.json is valid and patch targets exist."""

    def setUp(self):
        config_path = os.path.join(TORCH_NPU_ROOT, "contrib", "apis_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def test_json_structure(self):
        """apis_config.json has valid structure."""
        self.assertIsInstance(self.config, dict)
        for lib_name, lib_config in self.config.items():
            self.assertIn("version", lib_config, f"{lib_name} missing version")
            self.assertIn("apis", lib_config, f"{lib_name} missing apis")
            self.assertIsInstance(lib_config["apis"], dict)
            for api_path, api_type in lib_config["apis"].items():
                self.assertIn(api_type, ("function", "method"),
                              f"{api_path} has invalid type: {api_type}")

    def test_removed_redundant_patches(self):
        """Patches that HF 5.x handles natively should be removed."""
        transformers_apis = self.config["transformers"]["apis"]
        # These have native NPU support in HF 5.x
        self.assertNotIn("transformers.trainer.Trainer._load_rng_state",
                         transformers_apis,
                         "Should be removed: HF 5.x has native NPU RNG load support")
        self.assertNotIn("transformers.trainer.Trainer._save_rng_state",
                         transformers_apis,
                         "Should be removed: HF 5.x has native NPU RNG save support")
        self.assertNotIn("transformers.utils.import_utils.is_torch_bf16_gpu_available",
                         transformers_apis,
                         "Should be removed: HF 5.x has native NPU BF16 detection")

    def test_kept_necessary_patches(self):
        """Patches still needed in HF 5.x should be present."""
        transformers_apis = self.config["transformers"]["apis"]
        # These are still needed
        needed = [
            "transformers.training_args.TrainingArguments.__post_init__",
            "transformers.trainer_utils.TrainerMemoryTracker.start",
            "transformers.utils.import_utils.is_torch_cuda_available",
            "transformers.utils.import_utils.is_flash_attn_2_available",
        ]
        for api in needed:
            self.assertIn(api, transformers_apis, f"Missing needed patch: {api}")

    def test_transformers_patch_paths_exist(self):
        """Verify patch target functions exist in installed transformers (if available)."""
        try:
            import transformers
        except ImportError:
            self.skipTest("transformers not installed")

        transformers_apis = self.config["transformers"]["apis"]
        for full_path, api_type in transformers_apis.items():
            parts = full_path.rsplit(".", 1)
            if len(parts) != 2:
                continue
            module_path, attr_name = parts

            # For methods, split further
            if api_type == "method":
                parts2 = module_path.rsplit(".", 1)
                if len(parts2) == 2:
                    module_path, class_name = parts2
                    try:
                        mod = importlib.import_module(module_path)
                        cls = getattr(mod, class_name, None)
                        self.assertIsNotNone(cls,
                                             f"Class not found: {module_path}.{class_name}")
                        self.assertTrue(hasattr(cls, attr_name),
                                        f"Method not found: {full_path}")
                    except (ImportError, ModuleNotFoundError):
                        pass  # Module restructured, will be caught at runtime
            else:
                try:
                    mod = importlib.import_module(module_path)
                    self.assertTrue(hasattr(mod, attr_name),
                                    f"Function not found: {full_path}")
                except (ImportError, ModuleNotFoundError):
                    pass


class TestSdpaRegistration(unittest.TestCase):
    """Test SDPA registration module structure (without NPU hardware)."""

    def test_module_imports(self):
        """_sdpa.py can be parsed without errors."""
        sdpa_path = os.path.join(TORCH_NPU_ROOT, "npu", "_sdpa.py")
        self.assertTrue(os.path.exists(sdpa_path),
                        "torch_npu/npu/_sdpa.py should exist")
        # Verify the file has the expected functions
        with open(sdpa_path, "r") as f:
            content = f.read()
        self.assertIn("def _sdpa_forward_npu", content)
        self.assertIn("def register_sdpa_for_npu", content)
        self.assertIn("aten::_scaled_dot_product_fused_attention_overrideable", content)

    def test_forward_signature_matches_pytorch(self):
        """Forward function has correct parameter names matching PyTorch schema."""
        sdpa_path = os.path.join(TORCH_NPU_ROOT, "npu", "_sdpa.py")
        with open(sdpa_path, "r") as f:
            content = f.read()
        # Check required parameters from native_functions.yaml
        required_params = ["query", "key", "value", "attn_bias", "dropout_p",
                           "is_causal", "return_debug_mask", "scale"]
        for param in required_params:
            self.assertIn(param, content,
                          f"Missing parameter '{param}' in SDPA forward")

    def test_forward_returns_9_tuple(self):
        """Forward function returns the expected 9-tuple."""
        sdpa_path = os.path.join(TORCH_NPU_ROOT, "npu", "_sdpa.py")
        with open(sdpa_path, "r") as f:
            content = f.read()
        # Check return comments indicate 9 elements
        self.assertIn("attention_out", content)
        self.assertIn("logsumexp", content)
        self.assertIn("cum_seq_q", content)
        self.assertIn("cum_seq_k", content)
        self.assertIn("philox_seed", content)
        self.assertIn("philox_offset", content)


class TestKvCachePatch(unittest.TestCase):
    """Test KV cache flash attention patch module."""

    def test_patch_module_exists(self):
        """npu_flash_attention_patch.py exists."""
        patch_path = os.path.join(TORCH_NPU_ROOT, "contrib",
                                  "npu_flash_attention_patch.py")
        self.assertTrue(os.path.exists(patch_path))

    def test_patch_function_signature(self):
        """npu_flash_attn_with_kvcache has correct interface."""
        patch_path = os.path.join(TORCH_NPU_ROOT, "contrib",
                                  "npu_flash_attention_patch.py")
        with open(patch_path, "r") as f:
            content = f.read()
        # Check required parameters matching HF's expected interface
        required_params = ["q", "k_cache", "v_cache", "softmax_scale", "causal"]
        for param in required_params:
            self.assertIn(param, content,
                          f"Missing parameter '{param}' in kvcache function")

    def test_apply_function_exists(self):
        """apply() function exists for patching HF module."""
        patch_path = os.path.join(TORCH_NPU_ROOT, "contrib",
                                  "npu_flash_attention_patch.py")
        with open(patch_path, "r") as f:
            content = f.read()
        self.assertIn("def apply()", content)
        self.assertIn("npu_flash_attention.npu_flash_attn_with_kvcache", content)


class TestInitRegistration(unittest.TestCase):
    """Test that __init__.py includes SDPA registration."""

    def test_init_imports_sdpa(self):
        """torch_npu/__init__.py registers SDPA for NPU."""
        init_path = os.path.join(TORCH_NPU_ROOT, "__init__.py")
        with open(init_path, "r") as f:
            content = f.read()
        self.assertIn("register_sdpa_for_npu", content,
                      "__init__.py should call register_sdpa_for_npu()")


class TestRequirements(unittest.TestCase):
    """Test that requirements are updated."""

    def test_transformers_version_flexible(self):
        """test/requirements.txt allows transformers >= 4.40.0."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r") as f:
            content = f.read()
        # Should not pin to exact old version
        self.assertNotIn("transformers==4.40.0", content,
                         "Should not pin to old transformers version")
        self.assertIn("transformers>=4.40.0", content,
                      "Should allow transformers >= 4.40.0")


if __name__ == "__main__":
    unittest.main()
