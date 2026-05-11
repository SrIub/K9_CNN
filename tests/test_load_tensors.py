import unittest
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_tensors import make_tensors, make_splits

N_BINS = 10


def make_df(n_A4=0, n_A7=0, n_neg=0):
    """Build a DataFrame with K9 arrays and labels."""
    rows = []
    for _ in range(n_A4):
        rows.append({"A4_K9": np.ones(N_BINS,  dtype=np.float32) * 5,
                     "A7_K9": np.zeros(N_BINS, dtype=np.float32),
                     "label_A4": 1, "label_A7": 0})
    for _ in range(n_A7):
        rows.append({"A4_K9": np.zeros(N_BINS, dtype=np.float32),
                     "A7_K9": np.ones(N_BINS,  dtype=np.float32) * 3,
                     "label_A4": 0, "label_A7": 1})
    for _ in range(n_neg):
        rows.append({"A4_K9": np.zeros(N_BINS, dtype=np.float32),
                     "A7_K9": np.zeros(N_BINS, dtype=np.float32),
                     "label_A4": 0, "label_A7": 0})
    return pd.DataFrame(rows)

small_df = make_df(n_A4=2, n_A7=2, n_neg=2)
X, y = make_tensors(small_df)

class TestMakeTensors(unittest.TestCase):

    def test_X_shape(self):
        self.assertEqual(X.shape, (6, 1, 2, N_BINS))

    def test_y_shape(self):
        self.assertEqual(y.shape, (6, 2))

    def test_row_is_dim1_index0(self):
        # first 2 rows are A4 TEs with A4_K9=5
        self.assertAlmostEqual(X[0, 0, 0, 0].item(), 5.0)
        # rows 2-3 are A7 TEs with A7_K9=3
        self.assertAlmostEqual(X[2, 0, 1, 0].item(), 3.0)

    def test_y_labels_correct(self):
        # First row: A4 TE -> (1, 0)
        self.assertEqual(y[0, 0].item(), 1.0)
        self.assertEqual(y[0, 1].item(), 0.0)
        # Third row: A7 TE -> (0, 1)
        self.assertEqual(y[2, 0].item(), 0.0)
        self.assertEqual(y[2, 1].item(), 1.0)

# 20 rows per class = 60 total
full_df = make_df(n_A4=20, n_A7=20, n_neg=20)
train_df, val_df, test_df = make_splits(full_df, val_fraction=0.15, test_fraction=0.15, seed=42)

class TestMakeSplits(unittest.TestCase):

    def test_returns_three_dataframes(self):
        for split in (train_df, val_df, test_df):
            self.assertIsInstance(split, pd.DataFrame)

    def test_rows_sum_to_total(self):
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(full_df))

    def test_test_fraction_approximate(self):
        self.assertAlmostEqual(len(test_df) / len(full_df), 0.15, delta=0.03)

    def test_all_label_types_in_train(self):
        types = set(zip(train_df["label_A4"], train_df["label_A7"]))
        self.assertIn((1, 0), types)
        self.assertIn((0, 1), types)
        self.assertIn((0, 0), types)

    def test_no_row_appears_in_two_splits(self):
        idx_train = set(train_df.index)
        idx_val   = set(val_df.index)
        idx_test  = set(test_df.index)
        self.assertEqual(len(idx_train & idx_val),  0)
        self.assertEqual(len(idx_train & idx_test), 0)
        self.assertEqual(len(idx_val   & idx_test), 0)


if __name__ == "__main__":
    unittest.main()
