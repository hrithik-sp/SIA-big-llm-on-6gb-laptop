"""Tests for tools/task_eval.py: answer extraction and the exact sign test."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import task_eval  # noqa: E402


class SignTest(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(task_eval.sign_test(7, 3), 0.34375)  # 2 * (1+10+45+120) / 1024
        self.assertAlmostEqual(task_eval.sign_test(10, 0), 2 / 1024)
        self.assertEqual(task_eval.sign_test(0, 0), 1.0)
        self.assertEqual(task_eval.sign_test(4, 4), 1.0)

    def test_symmetric(self):
        self.assertEqual(task_eval.sign_test(2, 9), task_eval.sign_test(9, 2))


class Extract(unittest.TestCase):
    def test_letter(self):
        self.assertEqual(task_eval.extract_letter("so...\nAnswer: (C)"), "C")
        self.assertEqual(task_eval.extract_letter("The answer is **B**. Answer: D"), "D")
        self.assertIsNone(task_eval.extract_letter("no conclusion"))

    def test_number(self):
        self.assertEqual(task_eval.extract("work\n#### 1,234"), 1234)
        self.assertEqual(task_eval.extract("total is $5.50"), 5.5)


if __name__ == "__main__":
    unittest.main()
