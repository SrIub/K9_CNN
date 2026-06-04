import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import build_te_df

FLANK = 500   # half_flank=250, bins_per_side=5
N     = 10

# A4: flat 5 across all of 2L [0, 10000]
K9_A4 = {"2L": ([0], [10000], [5])}

# A7: 7 in [0, 3000], gap in [3000, 7000], 11 in [7000, 10000]
# Used to distinguish extraction position: src region gives 7, tgt region gives 11.
K9_A7 = {"2L": ([0, 7000], [3000, 10000], [7, 11])}

# A4 TE at 500-1500; A7 syntenic breakpoint at 8000 (inside the count=11 region).
# Old (buggy) code would extract A7 at A4's src position (500-1500 → count=7).
# Correct code extracts A7 at tgt=8000 → count=11.
A4_TO_A7 = [{"chrom": "2L", "src_start": 500, "src_end": 1500,
              "tgt_start": 8000, "tgt_end": 8000, "name": "roo", "family": "LTR/Bel-Pao"}]

# A7 TE at 7500-8500 (count=11 region); A4 breakpoint at 1000 (flat=5).
A7_TO_A4 = [{"chrom": "2L", "src_start": 7500, "src_end": 8500,
              "tgt_start": 1000, "tgt_end": 1000, "name": "doc", "family": "LINE/Jockey"}]

df = build_te_df(K9_A4, K9_A7, A4_TO_A7, A7_TO_A4, flank=FLANK, n_bins=N)


class TestBuildTeDf(unittest.TestCase):

    def test_row_count(self):
        self.assertEqual(len(df), 2)

    def test_columns(self):
        expected = {"chrom", "locus_start", "locus_end", "center", "A4_K9", "A7_K9", "label_A4", "label_A7"}
        self.assertEqual(set(df.columns), expected)

    def test_labels_a4_entry(self):
        self.assertEqual(df.iloc[0]["label_A4"], 1)
        self.assertEqual(df.iloc[0]["label_A7"], 0)

    def test_labels_a7_entry(self):
        self.assertEqual(df.iloc[1]["label_A4"], 0)
        self.assertEqual(df.iloc[1]["label_A7"], 1)

    def test_locus_coords_a4_entry(self):
        self.assertEqual(df.iloc[0]["locus_start"], 500)
        self.assertEqual(df.iloc[0]["locus_end"],   1500)
        self.assertEqual(df.iloc[0]["center"],      1000)

    def test_locus_coords_a7_entry(self):
        self.assertEqual(df.iloc[1]["locus_start"], 7500)
        self.assertEqual(df.iloc[1]["locus_end"],   8500)
        self.assertEqual(df.iloc[1]["center"],      8000)

    def test_K9_arrays_correct_shape(self):
        for i in range(len(df)):
            with self.subTest(row=i):
                self.assertEqual(df.iloc[i]["A4_K9"].shape, (N,))
                self.assertEqual(df.iloc[i]["A7_K9"].shape, (N,))

    def test_a4_k9_from_src(self):
        # A4 is flat=5, so both entries should have A4_K9=5 regardless of position
        np.testing.assert_allclose(df.iloc[0]["A4_K9"], 5.0)
        np.testing.assert_allclose(df.iloc[1]["A4_K9"], 5.0)

    def test_a7_extracted_at_syntenic_coord_not_src(self):
        # Row 0: A7 tgt is at 8000 (count=11). If extracted at src (500-1500) it would be 7.
        np.testing.assert_allclose(df.iloc[0]["A7_K9"], 11.0)

    def test_a7_k9_from_src(self):
        # Row 1: A7 src is at 7500-8500, both inside count=11 region
        np.testing.assert_allclose(df.iloc[1]["A7_K9"], 11.0)

    def test_missing_chrom_skipped(self):
        ann_2R = [{"chrom": "2R", "src_start": 100, "src_end": 200,
                   "tgt_start": 500, "tgt_end": 500, "name": "roo", "family": "LTR"}]
        df_miss = build_te_df(K9_A4, K9_A7, ann_2R, [], flank=FLANK, n_bins=N)
        self.assertEqual(len(df_miss), 0)

    def test_empty_annotations(self):
        df_empty = build_te_df(K9_A4, K9_A7, [], [], flank=FLANK, n_bins=N)
        self.assertEqual(len(df_empty), 0)


if __name__ == "__main__":
    unittest.main()
