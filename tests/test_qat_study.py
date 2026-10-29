"""QAT study helpers: exact per-tensor format overrides and the format check (no models or GPU needed)."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import qat_study as qs  # noqa: E402


class OverrideLines(unittest.TestCase):
    TYPES = {"blk.1.attn_q.weight": "Q4_0", "blk.11.attn_q.weight": "Q4_0", "token_embd.weight": "Q6_K",
             "per_layer_model_proj.weight": "F16", "blk.1.attn_norm.weight": "F32"}

    def test_each_pattern_matches_only_its_own_tensor(self):
        # llama-quantize searches each pattern in the tensor name (std::regex_search, ECMAScript)
        for line in qs.override_lines(self.TYPES):
            pattern, _, _ = line.rpartition("=")
            hits = [n for n in self.TYPES if re.search(pattern, n)]
            self.assertEqual(len(hits), 1, (pattern, hits))

    def test_types_are_lower_case_ggml_names(self):
        lines = dict(line.rpartition("=")[::2] for line in qs.override_lines(self.TYPES))
        self.assertEqual(lines[r"^token_embd\.weight$"], "q6_k")
        self.assertEqual(lines[r"^per_layer_model_proj\.weight$"], "f16")


class TypeMismatches(unittest.TestCase):
    def test_lossless_float_storage_is_not_a_mismatch(self):
        self.assertEqual(qs.type_mismatches({"a": "BF16"}, {"a": "F32"}), [])

    def test_different_quantization_is_a_mismatch(self):
        self.assertEqual(qs.type_mismatches({"a": "Q4_K", "b": "Q4_0"}, {"a": "Q6_K", "b": "Q4_0"}),
                         [("a", "Q4_K", "Q6_K")])

    def test_f16_versus_bf16_changes_values(self):
        self.assertEqual(qs.type_mismatches({"a": "BF16"}, {"a": "F16"}), [("a", "BF16", "F16")])

    def test_tensors_in_only_one_file_are_reported_separately(self):
        self.assertEqual(qs.type_mismatches({"a": "Q4_0", "extra": "Q4_0"}, {"a": "Q4_0"}), [])


class Paths(unittest.TestCase):
    def test_study_files_carry_the_size(self):
        # reference logits are named after the file, so two sizes must never share a name
        self.assertIn("gemma-4-E2B-O-bf16", qs.gguf_path("E2B", "O"))
        self.assertTrue(qs.gguf_path("E2B", "O4").endswith("gemma-4-E2B-O4-q4_0.gguf"))


if __name__ == "__main__":
    unittest.main()
