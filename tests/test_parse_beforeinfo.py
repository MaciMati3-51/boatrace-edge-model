import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parse_beforeinfo import ROOT, parse


class BeforeInfoTest(unittest.TestCase):
    def test_saved_real_page(self):
        path = ROOT / "data/raw/beforeinfo/202406/20240621_12_08.html"
        if not path.exists():
            self.skipTest("Downloaded sample not present")
        page = path.read_text()
        rows = {r["boat"]: r for r in parse(page, "20240621", 12, 8)}
        self.assertEqual(rows[6]["exhibition_course"], 5)
        self.assertEqual(rows[5]["exhibition_course"], 6)
        self.assertEqual(rows[6]["exhibition_st"], -0.02)
        self.assertEqual(rows[1]["exhibition_st"], 0.17)  # previous-race ST is .09
        self.assertEqual(rows[2]["parts_raw"], "シャフト")
        self.assertEqual(rows[4]["parts_raw"], "リング×4")
        with self.assertRaises(ValueError):
            parse(page.replace('table1_boatImage1Time', 'missing'), "20240621", 12, 8)

    def test_reject_empty_page(self):
        with self.assertRaises(ValueError):
            parse("<html>データはありません</html>", "20240621", 12, 8)
