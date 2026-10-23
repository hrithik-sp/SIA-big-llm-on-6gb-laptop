"""Parser test for tools/quality.py, using llama.cpp's own format strings
(tools/perplexity/perplexity.cpp, build 0a1ef309c)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import quality  # noqa: E402

SAMPLE = """\
llama_perplexity: some log line: 12.5
====== Perplexity statistics ======
Mean PPL(Q)                   :  10.123456 ±   0.123456
Mean PPL(base)                :  10.000000 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.50%
Mean ln(PPL(Q)/PPL(base))     :   0.012270 ±   0.001000
Mean PPL(Q)/PPL(base)         :   1.012346 ±   0.001012
Mean PPL(Q)-PPL(base)         :   0.123456 ±   0.010000

====== KL divergence statistics ======
Mean    KLD:   0.034567 ±   0.000456
Maximum KLD:   5.123456
99.9%   KLD:   1.234567
99.0%   KLD:   0.456789
Median  KLD:   0.012345
Minimum KLD:  -0.000001

====== Token probability statistics ======
Mean    Δp: -0.123 ± 0.045 %
RMS Δp    :  3.456 ± 0.078 %
Same top p: 93.210 ± 0.150 %
"""


class ParseTest(unittest.TestCase):
    def test_parses_llama_perplexity_kld_output(self):
        s = quality.parse(SAMPLE)
        self.assertAlmostEqual(s["Mean PPL(Q)"], 10.123456)
        self.assertAlmostEqual(s["Mean PPL(Q) ±"], 0.123456)
        self.assertAlmostEqual(s["Mean KLD"], 0.034567)
        self.assertAlmostEqual(s["Mean KLD ±"], 0.000456)
        self.assertAlmostEqual(s["99.0% KLD"], 0.456789)
        self.assertAlmostEqual(s["Minimum KLD"], -0.000001)
        self.assertAlmostEqual(s["Same top p"], 93.21)
        self.assertAlmostEqual(s["Same top p ±"], 0.15)
        self.assertAlmostEqual(s["RMS Δp"], 3.456)
        self.assertNotIn("llama_perplexity", " ".join(s))  # log lines before the stats are ignored

    def test_parses_base_run_final_estimate(self):
        s = quality.parse("[1]5.1,[2]6.2,\nFinal estimate: PPL = 7.6543 +/- 0.12345\n")
        self.assertEqual((s["Final PPL"], s["Final PPL ±"]), (7.6543, 0.12345))


if __name__ == "__main__":
    unittest.main()
