import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from load_df import load_K9, load_TE_map

TEST_K9  = Path(__file__).parent / "test_K9.bed"
TEST_MAP = Path(__file__).parent / "test_TE_map.txt"

k9     = load_K9(str(TEST_K9))
te_map = load_TE_map(str(TEST_MAP))

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


class TestLoadTEMap(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(te_map, list)

    def test_only_flag1_rows_kept(self):
        # fixture has 2 flag=1, 1 flag=0, 1 flag=-1
        self.assertEqual(len(te_map), 2)

    def test_keys(self):
        expected = {"chrom", "src_start", "src_end", "tgt_start", "tgt_end", "name", "family"}
        for rec in te_map:
            with self.subTest(rec=rec):
                self.assertEqual(set(rec.keys()), expected)

    def test_position_ints(self):
        for rec in te_map:
            with self.subTest(rec=rec):
                self.assertIsInstance(rec["src_start"], int)
                self.assertIsInstance(rec["src_end"],   int)
                self.assertIsInstance(rec["tgt_start"], int)
                self.assertIsInstance(rec["tgt_end"],   int)

    def test_first_record_values(self):
        rec = te_map[0]
        self.assertEqual(rec["chrom"],     "2L")
        self.assertEqual(rec["src_start"], 100)
        self.assertEqual(rec["src_end"],   200)
        self.assertEqual(rec["tgt_start"], 5000)
        self.assertEqual(rec["tgt_end"],   5000)
        self.assertEqual(rec["name"],      "roo")
        self.assertEqual(rec["family"],    "LTR/Bel-Pao")

    def test_second_record_tgt_range(self):
        rec = te_map[1]
        self.assertEqual(rec["tgt_start"], 8000)
        self.assertEqual(rec["tgt_end"],   8100)

    def test_empty_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# header\n")
            f.write("2L\t100\t200\troo\tLTR\t100\t0\t-1\t-1\n")
            fname = f.name
        result = load_TE_map(fname)
        os.unlink(fname)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
