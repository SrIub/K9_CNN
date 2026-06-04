import unittest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from simple_random_forest import extract_features, build_feature_matrix, BINS_PER_SIDE

KB_RANGE = list(range(1, 11))  # 1 through 10
EXPECTED_KEYS_PER_STRAIN = (
    [f"mean_{kb}kb_left"  for kb in KB_RANGE] +
    [f"mean_{kb}kb_right" for kb in KB_RANGE]
)
N_BINS = BINS_PER_SIDE * 2  # 400


def _make_array(left_val=0.0, right_val=0.0) -> np.ndarray:
    """Helper: flat array with left side = left_val, right side = right_val."""
    arr = np.zeros(N_BINS, dtype=np.float32)
    arr[:BINS_PER_SIDE]  = left_val
    arr[BINS_PER_SIDE:]  = right_val
    return arr


def _make_df(a4_array, a7_array) -> pd.DataFrame:
    """Helper: one-row DataFrame that mimics a te_df row."""
    return pd.DataFrame([{
        "chrom":       "2L",
        "locus_start": 1000,
        "locus_end":   2000,
        "center":      1500,
        "A4_K9":       a4_array,
        "A7_K9":       a7_array,
        "label_A4":    1,
        "label_A7":    0,
    }])


class TestExtractFeatures(unittest.TestCase):

    def test_returns_dict(self):
        arr = np.zeros(N_BINS, dtype=np.float32)
        result = extract_features(arr, "A4")
        self.assertIsInstance(result, dict)

    def test_key_names_with_prefix(self):
        arr = np.zeros(N_BINS, dtype=np.float32)
        result = extract_features(arr, "A4")
        for suffix in EXPECTED_KEYS_PER_STRAIN:
            self.assertIn(f"A4_{suffix}", result, msg=f"Missing key A4_{suffix}")

    def test_key_count(self):
        arr = np.zeros(N_BINS, dtype=np.float32)
        result = extract_features(arr, "A4")
        self.assertEqual(len(result), len(EXPECTED_KEYS_PER_STRAIN))

    def test_all_zeros_gives_zero_features(self):
        arr = np.zeros(N_BINS, dtype=np.float32)
        result = extract_features(arr, "A4")
        for key, val in result.items():
            with self.subTest(key=key):
                self.assertEqual(val, 0.0)

    def test_uniform_left_signal(self):
        # Left side = 5, right side = 0 → all left means = 5, all right means = 0
        arr = _make_array(left_val=5.0, right_val=0.0)
        result = extract_features(arr, "A4")
        for kb in KB_RANGE:
            with self.subTest(kb=kb):
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_left"],  5.0, places=5)
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_right"], 0.0, places=5)

    def test_uniform_right_signal(self):
        # Right side = 3, left side = 0 → all right means = 3, all left means = 0
        arr = _make_array(left_val=0.0, right_val=3.0)
        result = extract_features(arr, "A4")
        for kb in KB_RANGE:
            with self.subTest(kb=kb):
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_right"], 3.0, places=5)
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_left"],  0.0, places=5)

    def test_signal_only_in_1kb_left_window(self):
        # Only the last 20 bins of the left side (= 1kb from TE edge) are non-zero.
        # mean_Nkb_left = 10 * 20 / (N*20) = 10/N  (20 non-zero bins out of N*20 total)
        arr = np.zeros(N_BINS, dtype=np.float32)
        arr[180:200] = 10.0
        result = extract_features(arr, "A4")
        for kb in KB_RANGE:
            n_bins_in_window = kb * 20  # 20 bins per kb
            expected = 10.0 * 20 / n_bins_in_window
            with self.subTest(kb=kb):
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_left"], expected, places=5)
        self.assertAlmostEqual(result["A4_mean_1kb_right"], 0.0, places=5)

    def test_signal_only_in_1kb_right_window(self):
        # Only the first 20 bins of the right side (= 1kb from TE edge) are non-zero.
        # mean_Nkb_right = 8 * 20 / (N*20) = 8/N
        arr = np.zeros(N_BINS, dtype=np.float32)
        arr[200:220] = 8.0
        result = extract_features(arr, "A4")
        for kb in KB_RANGE:
            n_bins_in_window = kb * 20
            expected = 8.0 * 20 / n_bins_in_window
            with self.subTest(kb=kb):
                self.assertAlmostEqual(result[f"A4_mean_{kb}kb_right"], expected, places=5)
        self.assertAlmostEqual(result["A4_mean_1kb_left"], 0.0, places=5)

    def test_prefix_A7(self):
        arr = _make_array(left_val=2.0, right_val=4.0)
        result = extract_features(arr, "A7")
        for suffix in EXPECTED_KEYS_PER_STRAIN:
            self.assertIn(f"A7_{suffix}", result)
        # No A4 keys should appear
        self.assertFalse(any(k.startswith("A4_") for k in result))


class TestBuildFeatureMatrix(unittest.TestCase):

    def test_returns_dataframe(self):
        df = _make_df(_make_array(), _make_array())
        result = build_feature_matrix(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_row_count_matches_input(self):
        a4 = _make_array(left_val=1.0)
        a7 = _make_array(right_val=2.0)
        df = pd.concat([_make_df(a4, a7), _make_df(a7, a4)], ignore_index=True)
        result = build_feature_matrix(df)
        self.assertEqual(len(result), 2)

    def test_column_count(self):
        # 20 features per strain (10 left + 10 right) × 2 strains = 40 columns
        df = _make_df(_make_array(), _make_array())
        result = build_feature_matrix(df)
        self.assertEqual(result.shape[1], len(EXPECTED_KEYS_PER_STRAIN) * 2)

    def test_all_expected_columns_present(self):
        df = _make_df(_make_array(), _make_array())
        result = build_feature_matrix(df)
        for strain in ["A4", "A7"]:
            for suffix in EXPECTED_KEYS_PER_STRAIN:
                self.assertIn(f"{strain}_{suffix}", result.columns)

    def test_values_match_extract_features(self):
        a4 = _make_array(left_val=5.0, right_val=0.0)
        a7 = _make_array(left_val=0.0, right_val=3.0)
        df = _make_df(a4, a7)
        result = build_feature_matrix(df)
        for kb in KB_RANGE:
            with self.subTest(kb=kb):
                self.assertAlmostEqual(result.iloc[0][f"A4_mean_{kb}kb_left"],  5.0, places=5)
                self.assertAlmostEqual(result.iloc[0][f"A7_mean_{kb}kb_right"], 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
