import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_v3 as v3
import v3_features as feat


def synthetic(days=40, venues=(1, 2), races=6, seed=0):
    """小さな番組表・結果・レース情報。選手60人、F と欠場を少し混ぜる。"""
    rng = np.random.default_rng(seed)
    entries, results, meta = [], [], []
    start = date(2026, 1, 1)
    for d in range(days):
        day = (start + timedelta(days=d)).strftime("%Y%m%d")
        for jcd in venues:
            for race_no in range(1, races + 1):
                racers = rng.choice(60, 6, replace=False) + 4000
                ranks = rng.permutation(6) + 1
                flying = rng.random() < 0.05
                meta.append(dict(date=day, jcd=jcd, race_no=race_no, min_of_day=600 + 30 * race_no,
                                 race_type="予選", series="一般"))
                for b in range(6):
                    entries.append(dict(
                        date=day, jcd=jcd, race_no=race_no, deadline="10:00", boat=b + 1, racer_id=int(racers[b]),
                        name="x", age=30, branch="x", weight=52, **{"class": "B1"}, nat_wr=5.0, nat_2r=30.0,
                        loc_wr=5.0, loc_2r=30.0, motor_no=int(rng.integers(1, 20)),
                        motor_2r=0.0 if d == 20 else 30.0, boat_no=int(rng.integers(1, 20)), boat_2r=30.0,
                        setsu_seiseki=""))
                    raw = "F" if flying and b == 5 else f"0{ranks[b]}"
                    results.append(dict(
                        date=day, jcd=jcd, race_no=race_no, distance=1800, weather="晴", wind_dir="北",
                        wind_speed=1, wave=1, kimarite="逃げ" if ranks[0] == 1 else "まくり", rank_raw=raw,
                        boat=b + 1, racer_id=int(racers[b]), motor_no=entries[-1]["motor_no"],
                        boat_no=entries[-1]["boat_no"], tenji=float(np.round(6.7 + rng.random() * .3, 2)),
                        entry_course=b + 1, st=f"0.{rng.integers(5, 30):02d}", racetime="1.50.0"))
    return pl.DataFrame(entries), pl.DataFrame(results), pl.DataFrame(meta)


class FeatureTest(unittest.TestCase):
    def test_features_ignore_results_on_and_after_the_race_date(self):
        entries, results, meta = synthetic()
        target = "20260130"
        full = feat.build(entries, results, meta).filter(pl.col("date") == target)
        tenji = results.filter(pl.col("date") == target).select(*feat.BK, "tenji")
        live = feat.build(entries.filter(pl.col("date") <= target), results.filter(pl.col("date") < target),
                          meta, tenji=tenji).filter(pl.col("date") == target)
        self.assertEqual(full.height, live.height)
        for c in feat.per_boat_columns(full) + feat.RACE_LEVEL:
            a, b = full[c].to_numpy().astype(float), live[c].to_numpy().astype(float)
            np.testing.assert_allclose(a, b, equal_nan=True, err_msg=c)

    def test_history_counts_only_earlier_days(self):
        entries, results, meta = synthetic()
        df = feat.build(entries, results, meta)
        first_day = df.filter(pl.col("date") == "20260101")
        self.assertEqual(first_day["r_n_all"].max(), 0)  # 初日は履歴なし
        racer = entries["racer_id"][0]
        row = df.filter((pl.col("racer_id") == racer) & (pl.col("date") >= "20260120")).sort("date").row(0, named=True)
        earlier = results.filter((pl.col("racer_id") == racer) & (pl.col("date") < row["date"])).height
        self.assertEqual(row["r_n_all"], earlier)

    def test_motor_period_restarts_when_all_rates_reset(self):
        entries, _, _ = synthetic()
        periods = feat.motor_periods(entries).filter(pl.col("jcd") == 1).sort("date")
        self.assertEqual(periods.filter(pl.col("date") < "20260121")["mperiod"].n_unique(), 1)
        self.assertEqual(periods["mperiod"].n_unique(), 2)


class ModelTest(unittest.TestCase):
    def test_probabilities_are_consistent(self):
        entries, results, meta = synthetic(days=60)
        df = feat.build(entries, results, meta).join(feat.labels(results), on=feat.BK + ["racer_id"], how="left")
        df = df.filter(pl.col("rank").is_not_null()).sort(feat.BK)
        model = v3.RaceModelV3().fit(df, iterations=dict(cl1=3, flat1=3, cl2=3), log=lambda *_: None)
        p1, joint = model.predict(df.filter(pl.col("date") >= "20260220"))
        self.assertEqual(joint.shape[1], 30)
        np.testing.assert_allclose(p1.sum(axis=1), 1)
        np.testing.assert_allclose(joint.sum(axis=1), 1)
        # 1着ごとの周辺は1着確率に一致する（1着艇は2着にならない）
        for a in range(6):
            np.testing.assert_allclose(joint[:, v3.PAIRS[:, 0] == a].sum(axis=1), p1[:, a], rtol=1e-6)
        self.assertTrue((joint > 0).all())

    def test_group_softmax_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=12)
        y = np.zeros(12)
        y[[2, 9]] = 1

        class Data:
            def get_label(self):
                return y

        grad, _ = v3.objective(6)(scores, Data())
        loss = lambda s: -np.log((v3.group_softmax(s, 6) * y.reshape(-1, 6)).sum(axis=1)).sum()
        for i in range(12):
            e = np.zeros(12)
            e[i] = 1e-6
            self.assertAlmostEqual(grad[i], (loss(scores + e) - loss(scores - e)) / 2e-6, places=5)


if __name__ == "__main__":
    unittest.main()
