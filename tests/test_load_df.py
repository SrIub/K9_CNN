import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import load_K9, load_TE_ann

TEST_K9  = Path(__file__).parent / "test_K9.bed"
TEST_ANN = Path(__file__).parent / "test_TE_ann.txt"

k9     = load_K9(str(TEST_K9))
te_ann = load_TE_ann(str(TEST_ANN))

class TestLoadK9(unittest.TestCase):

    def test_correct_chromosomes_loaded(self):
        self.assertEqual(set(k9.keys()), {"2L", "2R", "X", "3L", "3R"})

    def test_K9_dict_structure(self):
        for chrom, (starts, ends, counts) in k9.items():
            with self.subTest(chrom=chrom):
                self.assertIsInstance(starts, list)
                self.assertIsInstance(ends, list)
                self.assertIsInstance(counts, list)
                self.assertEqual(len(starts), len(ends))
                self.assertEqual(len(starts), len(counts))

    def test_values_are_ints(self):
        for chrom, (starts, ends, counts) in k9.items():
            with self.subTest(chrom=chrom):
                self.assertTrue(all(isinstance(x, int) for x in starts))
                self.assertTrue(all(isinstance(x, int) for x in ends))
                self.assertTrue(all(isinstance(x, int) for x in counts))

    def test_2L_has_four_intervals(self):
        self.assertEqual(len(k9["2L"][0]), 4)

    def test_2L_correct_values(self):
        starts, ends, counts = k9["2L"]
        self.assertEqual(starts, [0, 50460, 51984, 52061])
        self.assertEqual(ends,   [50407, 51982, 51986, 52093])
        self.assertEqual(counts, [0, 0, 5, 14])

    def test_2R_single_interval(self):
        starts, ends, counts = k9["2R"]
        self.assertEqual(starts, [50407])
        self.assertEqual(ends,   [50460])
        self.assertEqual(counts, [1])

    def test_3R_sorted(self):
        starts, ends, counts = k9["3R"]
        self.assertEqual(starts, [200, 51993])
        self.assertEqual(ends,   [300, 52052])
        self.assertEqual(counts, [17, 11])

    def test_starts_are_sorted_for_all_chroms(self):
        for chrom, (starts, *_) in k9.items():
            with self.subTest(chrom=chrom):
                self.assertEqual(starts, sorted(starts))


class TestLoadTEAnn(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(te_ann, list)

    def test_number_of_tes(self):
        self.assertEqual(len(te_ann), 4)

    def test_keys(self):
        for rec in te_ann:
            with self.subTest(rec=rec):
                self.assertIn("chrom",  rec)
                self.assertIn("start",  rec)
                self.assertIn("end",    rec)
                self.assertIn("name",   rec)
                self.assertIn("family", rec)

    def test_no_extra_keys(self):
        for rec in te_ann:
            self.assertEqual(set(rec.keys()), {"chrom", "start", "end", "name", "family"})

    def test_position_int(self):
        for rec in te_ann:
            with self.subTest(rec=rec):
                self.assertIsInstance(rec["start"], int)
                self.assertIsInstance(rec["end"],   int)

    def test_first_record_values(self):
        rec = te_ann[0]
        self.assertEqual(rec["chrom"],  "2L")
        self.assertEqual(rec["start"],  100)
        self.assertEqual(rec["end"],    200)
        self.assertEqual(rec["name"],   "roo")
        self.assertEqual(rec["family"], "LTR/Bel-Pao")

    def test_extra_columns_ignored(self):
        rec = te_ann[0]
        self.assertEqual(len(rec), 5)


if __name__ == "__main__":
    unittest.main()
