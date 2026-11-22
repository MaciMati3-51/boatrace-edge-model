import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_v4 as v4
import v3_features as feat
import v4_ratings as rt
from test_live_v3 import synthetic


class MetaTest(unittest.TestCase):
    def test_fixed_entry_flag_and_race_type(self):
        import tempfile
        import v4_meta
        lines = ["24BBGN", "ボートレース大　村   　７月　１日  日刊スポーツ杯　　　  第　２日", "",
                 "　１Ｒ  予　選　　　          Ｈ１８００ｍ  電話投票締切予定１５：１５",
                 "　５Ｒ  予　選　　　 進入固定 Ｈ１８００ｍ  電話投票締切予定１４：５３",
                 "　６Ｒ  進入固定戦隊 進入固定 Ｈ１８００ｍ  電話投票締切予定１５：３０",
                 "　７Ｒ  一般　　　　          Ｈ１８００ｍ  電話投票締切予定１６：００",
                 "１２Ｒ  優勝戦　　　          Ｈ１８００ｍ  電話投票締切予定２０：３０", "24BEND"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "B260701.TXT"
            path.write_bytes("\r\n".join(lines).encode("cp932"))
            rows = {r["race_no"]: r for r in v4_meta.parse_file(path)}
        self.assertEqual({k: (v["fixed4"], v["rtype4"]) for k, v in rows.items()},
                         {1: (0, 2), 5: (1, 2), 6: (1, 0), 7: (0, 1), 12: (0, 5)})
        self.assertEqual(rows[1]["date"], "20260701")
        self.assertEqual(rows[1]["jcd"], 24)


class RatingTest(unittest.TestCase):
    def test_ratings_ignore_results_on_and_after_the_race_date(self):
        entries, results, _ = synthetic()
        target = "20260130"
        full = rt.build(entries, results).filter(pl.col("date") == target)
        live = rt.build(entries.filter(pl.col("date") <= target),
                        results.filter(pl.col("date") < target)).filter(pl.col("date") == target)
        self.assertEqual(full.height, live.height)
        for c in rt.COLUMNS:
            np.testing.assert_allclose(full[c].to_numpy(), live[c].to_numpy(), err_msg=c)

    def test_first_day_is_zero_and_probabilities_sum_to_one(self):
        entries, results, _ = synthetic()
        df = rt.build(entries, results)
        first = df.filter(pl.col("date") == "20260101")
        self.assertEqual(first["rt_s_theta"].abs().max(), 0.0)
        self.assertEqual(first["rt_n"].max(), 0.0)
        total = df.group_by(rt.KEY).agg(pl.col("rt_s_p").sum(), pl.col("rt_f_p").sum())
        np.testing.assert_allclose(total["rt_s_p"].to_numpy(), 1.0)
        np.testing.assert_allclose(total["rt_f_p"].to_numpy(), 1.0)

    def test_stronger_racer_gets_higher_rating(self):
        # 選手 4000 が常に1着になるように結果を書き換える
        entries, results, _ = synthetic(days=30)
        has = results.filter(pl.col("racer_id") == 4000).select(rt.KEY).unique()
        fixed = results.join(has, on=rt.KEY, how="semi").with_columns(
            rank_raw=pl.when(pl.col("racer_id") == 4000).then(pl.lit("01")).otherwise(
                pl.format("0{}", pl.col("boat").rank("ordinal").over(rt.KEY) % 5 + 2)))
        results = pl.concat([results.join(has, on=rt.KEY, how="anti"), fixed.select(results.columns)])
        df = rt.build(entries, results).join(entries.select(*rt.BK, "racer_id"), on=rt.BK)
        last = df.sort("date").group_by("racer_id").agg(pl.col("rt_s_racer").last())
        best = last.sort("rt_s_racer", descending=True)["racer_id"][0]
        self.assertEqual(best, 4000)

    def test_flying_and_absent_boats_do_not_move_ratings(self):
        entries, results, _ = synthetic(days=5)
        only_f = results.with_columns(rank_raw=pl.lit("F"))
        df = rt.build(entries, only_f)
        self.assertEqual(df["rt_s_racer"].abs().max(), 0.0)
        self.assertEqual(df["rt_n"].max(), 0.0)


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entries, results, meta = synthetic(days=60)
        df = feat.build(entries, results, meta).join(feat.labels(results), on=feat.BK + ["racer_id"], how="left")
        df = df.filter(pl.col("rank").is_not_null())
        full = df.group_by(feat.KEY).len().filter(pl.col("len") == 6).select(feat.KEY)
        meta4 = meta.select(*feat.KEY, fixed4=(pl.col("race_no") == 5).cast(pl.Int64), rtype4=pl.lit(2))
        cls.df = v4.add_ratings(df.join(full, on=feat.KEY).sort(feat.BK), entries, results, meta4)
        cls.model = v4.RaceModelV4().fit(cls.df, iterations=dict(cl1=3, flat1=3, cl2=3, cl3=3), log=lambda *_: None)
        cls.test = cls.df.filter(pl.col("date") >= "20260220")
        cls.p1, cls.joint, cls.tri = cls.model.predict(cls.test)

    def test_probabilities_are_consistent(self):
        p1, joint, tri = self.p1, self.joint, self.tri
        self.assertEqual((joint.shape[1], tri.shape[1]), (30, 120))
        np.testing.assert_allclose(p1.sum(axis=1), 1)
        np.testing.assert_allclose(joint.sum(axis=1), 1)
        np.testing.assert_allclose(tri.sum(axis=1), 1)
        self.assertTrue((tri > 0).all())
        # 3連単を3着で足し上げると2連単、2連単を2着で足し上げると1着に一致する
        for j, (a, b) in enumerate(v4.PAIRS):
            sel = (v4.TRIPLES[:, 0] == a) & (v4.TRIPLES[:, 1] == b)
            np.testing.assert_allclose(tri[:, sel].sum(axis=1), joint[:, j], rtol=1e-6)
        for a in range(6):
            np.testing.assert_allclose(joint[:, v4.PAIRS[:, 0] == a].sum(axis=1), p1[:, a], rtol=1e-6)
        np.testing.assert_allclose(v4.trio_from_trifecta(tri).sum(axis=1), 1)

    def test_one_seed_per_tree_model_and_average(self):
        self.assertEqual(len(self.model.models["cl1"]), len(v4.SEEDS))
        self.assertEqual(len(self.model.models["flat1"]), 1)

    def test_losses_pick_the_realised_combination(self):
        order = np.argsort(self.test["rank"].to_numpy().reshape(-1, 6), axis=1)
        ls = v4.losses(order, self.p1, self.joint, self.tri)
        self.assertEqual(ls.shape, (len(order), 5))
        i = 0
        a, b, c = order[i, :3]
        k = [j for j, t in enumerate(v4.TRIPLES) if tuple(t) == (a, b, c)][0]
        self.assertAlmostEqual(ls[i, 3], -np.log(self.tri[i, k]))
        perms = [j for j, t in enumerate(v4.TRIPLES) if sorted(t) == sorted((a, b, c))]
        self.assertAlmostEqual(ls[i, 4], -np.log(self.tri[i, perms].sum()))
        # 3連複の的中確率 ≥ 3連単の的中確率
        self.assertTrue((ls[:, 4] <= ls[:, 3] + 1e-12).all())

    def test_saved_model_predicts_the_same(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            v4.save_model(self.model, path)
            p1, joint, tri = v4.load_model(path).predict(self.test)
        np.testing.assert_allclose(tri, self.tri)

    def test_rating_columns_reach_the_trees(self):
        d = v4.Design(self.test)
        for c in ("rt_s_theta", "rt_s_theta_rel", "rt_f_theta_rk", "rt_s_p", "rt_s_theta_in", "fixed4", "rtype4"):
            self.assertIn(c, d.boat_names)
        # レース単位の列は6クラス側に1回だけ入る
        self.assertEqual(sum(n.startswith("fixed4") for n in d.flat_names), 1)
        _, names, _ = d.third_rows(np.arange(2), np.array([0, 1]), np.array([1, 0]))
        self.assertIn("rt_s_theta_dS", names)


if __name__ == "__main__":
    unittest.main()
