import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import build_te_df, sample_negatives, build_full_df

W = 1000
N = 10

# K9: 4 intervals on 2L with known midpoints: 2500, 7500, 12500, 17500
K9_A4 = {"2L": ([0, 5000, 10000, 15000], [5000, 10000, 15000, 20000], [5, 3, 4, 2])}
K9_A7 = {"2L": ([0, 5000, 10000, 15000], [5000, 10000, 15000, 20000], [3, 2, 5, 1])}

# One A4 TE at center 2500, window=1000 -> TE window stored as (1500, 3500)
# Overlap check excludes any candidate whose window overlaps that, i.e. |candidate-2500| < 2*W
# Candidate 2500 is excluded; 7500, 12500, 17500 are not
ANN_A4 = [{"chrom": "2L", "start": 2000, "end": 3000, "name": "roo", "family": "LTR"}]

te_df  = build_te_df(K9_A4, K9_A7, ANN_A4, [], window=W, n_bins=N)
neg_df = sample_negatives(te_df, K9_A4, K9_A7, neg_ratio=1.0, window=W, n_bins=N, seed=42)


class TestSampleNegatives(unittest.TestCase):

    def test_row_count(self):
        self.assertEqual(len(neg_df), 1)

    def test_neg_labels(self):
        self.assertEqual(neg_df.iloc[0]["label_A4"], 0)
        self.assertEqual(neg_df.iloc[0]["label_A7"], 0)

    def test_columns_match_te_df(self):
        self.assertEqual(set(neg_df.columns), set(te_df.columns))

    def test_K9_arrays_correct_shape(self):
        self.assertEqual(neg_df.iloc[0]["A4_K9"].shape, (N,))
        self.assertEqual(neg_df.iloc[0]["A7_K9"].shape, (N,))

    def test_negative_not_near_te(self):
        center = neg_df.iloc[0]["center"]
        te_center = te_df.iloc[0]["center"]
        self.assertGreater(abs(center - te_center), 2 * W)

    def test_no_valid_candidates(self):
        k9_tiny = {"2L": ([2000], [3000], [5])}
        neg = sample_negatives(te_df, k9_tiny, k9_tiny, neg_ratio=1.0, window=W, n_bins=N, seed=42)
        self.assertEqual(len(neg), 0)


class TestBuildFullDf(unittest.TestCase):

    def setUp(self):
        self.full_df = build_full_df(te_df, neg_df)

    def test_row_count(self):
        self.assertEqual(len(self.full_df), len(te_df) + len(neg_df))

    def test_columns_preserved(self):
        self.assertEqual(set(self.full_df.columns), set(te_df.columns))

    def test_index_is_reset(self):
        self.assertEqual(list(self.full_df.index), list(range(len(self.full_df))))


if __name__ == "__main__":
    unittest.main()
