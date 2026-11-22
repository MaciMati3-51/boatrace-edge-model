import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exacta_settlement as settlement


class ExactaSettlementTest(unittest.TestCase):
    def test_minimum_compatible_votes_on_uniform_board(self):
        for counts in [np.ones(30, dtype=int), np.arange(1, 31)]:
            odds = (15 * counts.sum() // (2 * counts)) / 10
            self.assertEqual(settlement.implied_votes(odds), counts.sum())
        self.assertTrue(np.isnan(settlement.implied_votes(np.ones(30))))

    def test_score_v4_output_in_pair_order_and_preserve_baseline(self):
        coef = json.loads(settlement.CALIBRATION.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(json.dumps(dict(model_sha256=coef["model_sha256"],
                                                               calibration={"2t": {"combined": coef["combined"]}})))
            rows = [dict(date="20270113", jcd=1, race_no=1, pool="2t", first=a, second=b,
                         odds=22.5, p_final=1/30, ev_settle=.75, paper_candidate=False,
                         captured_at="2027-01-13T12:00:00+09:00", deadline_at="2027-01-13T12:09:00+09:00")
                    for a, b in settlement.m2.POOLS["2t"]]
            pl.DataFrame(rows[::-1]).write_csv(root / "final.csv")
            out = root / "candidate.csv"
            settlement.score(root, out)
            df = pl.read_csv(out)
            self.assertEqual(df["estimated_votes"].to_list(), [30] * 30)
            self.assertEqual(df.select("first", "second").rows(), settlement.m2.POOLS["2t"])
            np.testing.assert_allclose(df["ev_settle"], df["p_final"] * df["expected_settled_odds"])
            self.assertEqual(df["paper_candidate"].to_list(), (df["ev_settle"] > 1).to_list())
            self.assertTrue((df["v4_ev_settle"] == .75).all())
            with self.assertRaises(FileExistsError):
                settlement.score(root, out)
