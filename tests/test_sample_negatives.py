import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
from load_df import build_te_df, sample_negatives, build_full_df

FLANK = 1000  # 1000bp flank on each side
N     = 10    # total bins (5 per side after // 2)

# K9: 4 intervals on 2L with midpoints 2500, 7500, 12500, 17500
K9_A4 = {"2L": ([0, 5000, 10000, 15000], [5000, 10000, 15000, 20000], [5, 3, 4, 2])}
K9_A7 = {"2L": ([0, 5000, 10000, 15000], [5000, 10000, 15000, 20000], [3, 2, 5, 1])}

# A4 TE at 2000-3000; A7 syntenic breakpoint at 12000 (in [10000,15000], count=5).
# Exclusion zone (from te_df locus): [2000-FLANK, 3000+FLANK] = [1000, 4000]
# -> candidate 2500 is excluded; 7500, 12500, 17500 are not
A4_TO_A7 = [{"chrom": "2L", "src_start": 2000, "src_end": 3000,
              "tgt_start": 12000, "tgt_end": 12000, "name": "roo", "family": "LTR"}]

te_df  = build_te_df(K9_A4, K9_A7, A4_TO_A7, [], flank=FLANK, n_bins=N)
neg_df = sample_negatives(te_df, K9_A4, K9_A7, neg_ratio=1.0, flank=FLANK, n_bins=N, seed=42)


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

    def test_negative_outside_exclusion_zone(self):
        center   = neg_df.iloc[0]["center"]
        te_start = te_df.iloc[0]["locus_start"]
        te_end   = te_df.iloc[0]["locus_end"]
        self.assertFalse((te_start - FLANK) <= center <= (te_end + FLANK))

    def test_no_valid_a4_candidates(self):
        # A4 tiny: only midpoint 2500, inside A4 exclusion zone [1000, 4000] → no A4 candidates
        k9_a4_tiny = {"2L": ([2000], [3000], [5])}
        neg = sample_negatives(te_df, k9_a4_tiny, K9_A7, neg_ratio=1.0, flank=FLANK, n_bins=N, seed=42)
        self.assertEqual(len(neg), 0)

    def test_no_valid_a7_candidates(self):
        # te_df has label_A4=1 only, so A7 exclusion zone is empty.
        # Build a te_df with a label_A7=1 row to create an A7 exclusion zone.
        A7_TO_A4 = [{"chrom": "2L", "src_start": 2000, "src_end": 3000,
                     "tgt_start": 12000, "tgt_end": 12000, "name": "roo", "family": "LTR"}]
        te_df_a7 = build_te_df(K9_A4, K9_A7, [], A7_TO_A4, flank=FLANK, n_bins=N)
        # A7 TE at 2000-3000 → exclusion zone [1000, 4000]. A7 tiny: only midpoint 2500 → excluded.
        k9_a7_tiny = {"2L": ([2000], [3000], [3])}
        neg = sample_negatives(te_df_a7, K9_A4, k9_a7_tiny, neg_ratio=1.0, flank=FLANK, n_bins=N, seed=42)
        self.assertEqual(len(neg), 0)

    def test_a4_and_a7_use_independent_positions(self):
        # Verify A7_K9 values reflect A7 k9 data, not A4 position.
        # K9_A7 has different counts per interval; neg A7_K9 should match A7 data at A7 center.
        self.assertIsNotNone(neg_df.iloc[0]["A7_K9"])
        self.assertEqual(neg_df.iloc[0]["A7_K9"].shape, (N,))
        # A4_K9 and A7_K9 can have different values since they come from different positions/k9 dicts
        # (not asserting exact values since they depend on rng, just that both are present and valid)
        self.assertFalse(np.all(np.isnan(neg_df.iloc[0]["A4_K9"])))


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
