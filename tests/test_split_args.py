"""tools/bench.split_args: Windows paths keep their backslashes (the spec-gptoss-v1 bug)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import bench  # noqa: E402


class SplitArgs(unittest.TestCase):
    def test_windows_path_kept(self):
        self.assertEqual(bench.split_args(r"--spec-type draft-eagle3 -md C:\Users\x\m.gguf --spec-draft-n-max 2"),
                         ["--spec-type", "draft-eagle3", "-md", r"C:\Users\x\m.gguf", "--spec-draft-n-max", "2"])

    def test_quotes_group_and_are_removed(self):
        self.assertEqual(bench.split_args(r'-md "C:\My Models\d.gguf" -ot "blk\.1\.ffn.*=CPU"'),
                         ["-md", r"C:\My Models\d.gguf", "-ot", r"blk\.1\.ffn.*=CPU"])

    def test_empty(self):
        self.assertEqual(bench.split_args(""), [])


if __name__ == "__main__":
    unittest.main()
