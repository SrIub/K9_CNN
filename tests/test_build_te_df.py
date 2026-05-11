import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import build_te_df

W = 1000
N = 10

# Flat K9 signal across [0, 10000] on 2L
K9_A4 = {"2L": ([0], [10000], [5])}
K9_A7 = {"2L": ([0], [10000], [3])}

# One A4 TE on 2L: center = 1000
ANN_A4 = [{"chrom": "2L", "start": 500, "end": 1500, "name": "roo", "family": "LTR/Bel-Pao"}]

# One A7 TE on 2L: center = 5000
ANN_A7 = [{"chrom": "2L", "start": 4500, "end": 5500, "name": "doc", "family": "LINE/Jockey"}]

df = build_te_df(K9_A4, K9_A7, ANN_A4, ANN_A7, window=W, n_bins=N)


class TestBuildTeDf(unittest.TestCase):

    def test_row_count(self):
        self.assertEqual(len(df), 2)

    def test_columns(self):
        expected = {"chrom", "locus_start", "locus_end", "center", "A4_K9", "A7_K9", "label_A4", "label_A7"}
        self.assertEqual(set(df.columns), expected)

    def test_strain_labels(self):
        self.assertEqual(df.iloc[0]["label_A4"], 1) # TE on A4
        self.assertEqual(df.iloc[0]["label_A7"], 0)
        self.assertEqual(df.iloc[1]["label_A4"], 0) # TE on A7
        self.assertEqual(df.iloc[1]["label_A7"], 1)

    def test_center_calculation(self):
        self.assertEqual(df.iloc[0]["center"], 1000)
        self.assertEqual(df.iloc[1]["center"], 5000)

    def test_locus_start_end_preserved(self):
        self.assertEqual(df.iloc[0]["locus_start"], 500)
        self.assertEqual(df.iloc[0]["locus_end"],   1500)

    def test_K9_arrays_correct_shape(self):
        for i in range(len(df)):
            with self.subTest(row=i):
                self.assertEqual(df.iloc[i]["A4_K9"].shape, (N,))
                self.assertEqual(df.iloc[i]["A7_K9"].shape, (N,))

    def test_missing_chrom(self):
        ann_2R = [{"chrom": "2R", "start": 100, "end": 200, "name": "roo", "family": "LTR"}]
        df_miss = build_te_df(K9_A4, K9_A7, ann_2R, [], window=W, n_bins=N)
        self.assertEqual(len(df_miss), 0)

    def test_empty_annotations(self):
        df_empty = build_te_df(K9_A4, K9_A7, [], [], window=W, n_bins=N)
        self.assertEqual(len(df_empty), 0)


if __name__ == "__main__":
    unittest.main()
