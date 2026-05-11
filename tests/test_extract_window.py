import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import extract_K9_window

# Fixed paramaters for all tests:
#   window=1000, n_bins=10, center=1000
#   -> bin_size=200 bp, query=[0, 2000)
#   -> bin 0=[0,200), bin 1=[200,400), ..., bin 4=[800,1000),
#      bin 5=[1000,1200), ..., bin 9=[1800,2000)

W = 1000
N = 10
C = 1000


class TestExtractK9Window(unittest.TestCase):

    def test_missing_chrom(self):
        k9 = {"2R": ([0], [200], [5])}
        self.assertIsNone(extract_K9_window(k9, "2L", C, W, N))

    def test_output_shape(self):
        k9 = {"2L": ([0], [2000], [1])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertEqual(result.shape, (N,))

    def test_output_type_float(self):
        k9 = {"2L": ([0], [2000], [1])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertEqual(result.dtype, np.float32)

    def test_interval_spanning_full_query_fills_all_bins(self):
        # [0, 2000] with count=5 covers every bin completely
        k9 = {"2L": ([0], [2000], [5])}
        result = extract_K9_window(k9, "2L", C, W, N)
        np.testing.assert_allclose(result, 5.0)

    def test_single_interval_in_one_bin(self):
        # [0, 200] covers only bin 0
        k9 = {"2L": ([0], [200], [10])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertAlmostEqual(result[0], 10.0, places=4)
        self.assertEqual((result > 0).sum(), 1)

    def test_spike_splits_evenly(self):
        # [900, 1100] overlaps bin 4 [800,1000) and bin 5 [1000,1200)
        # 100 bp in each bin -> weight 0.5 each -> count 20 * 0.5 = 10 per bin
        k9 = {"2L": ([900], [1100], [20])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertAlmostEqual(result[4], 10.0, places=4)
        self.assertAlmostEqual(result[5], 10.0, places=4)
        self.assertAlmostEqual(result.sum(), 20.0, places=4)

    def test_interval_outside_query(self):
        k9 = {"2L": ([5000], [6000], [10])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertEqual(result.sum(), 0.0)

    def test_interval_partially_outside_query(self):
        # [-100, 100] overlaps query by 100 bp in bin 0 [0, 200)
        # weight = 100/200 = 0.5 -> result[0] = 10 * 0.5 = 5.0
        k9 = {"2L": ([-100], [100], [10])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertAlmostEqual(result[0], 5.0, places=4)
        self.assertEqual((result > 0).sum(), 1)

    def test_zero_count_interval(self):
        k9 = {"2L": ([0], [2000], [0])}
        result = extract_K9_window(k9, "2L", C, W, N)
        self.assertEqual(result.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
