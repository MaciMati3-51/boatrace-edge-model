import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_v2 as v2
import live_market_v2 as market


def fixture(date="20260912"):
    rows = []
    for b in range(1, 7):
        row = dict.fromkeys(v2.ENTRY, 1.0)
        row.update(date=date, jcd=18, race_no=4, boat=b, racer_id=4000 + b, class_ord=2,
                   tenji=6.80 + .01 * b, tkz_captured_at=f"{date[:4]}-{date[4:6]}-{date[6:]}T12:08:00+09:00",
                   deadline_at=f"{date[:4]}-{date[4:6]}-{date[6:]}T12:17:00+09:00")
        rows.append(row)
    return pl.DataFrame(rows)


def realistic_board(pool, date="20260912", jcd=18, race_no=4, captured="2026-09-12T12:08:00+09:00",
                    deadline="2026-09-12T12:17:00+09:00"):
    """Σ(1/オッズ) が控除率どおり約1.34になる板。"""
    pairs = market.POOLS[pool]
    q = np.linspace(3, 1, len(pairs))
    q /= q.sum()
    odds = np.floor(10 / (1.3407 * q)) / 10
    return pl.DataFrame([dict(date=date, jcd=jcd, race_no=race_no, pool=pool, first=a, second=b,
                              odds=float(o), captured_at=captured, deadline_at=deadline)
                         for (a, b), o in zip(pairs, odds)])


class InputTest(unittest.TestCase):
    def test_validate_accepts_extra_columns_and_rejects_bad_rows(self):
        df = fixture().with_columns(tilt=pl.lit(0.5), exhibition_st=pl.lit(.1))
        v2.validate_input(df)
        for bad in [df.head(5),
                    df.with_columns(tkz_captured_at=pl.lit("2026-09-12T12:17:00+09:00")),
                    df.with_columns(tkz_captured_at=pl.lit("2026-09-12T11:00:00+09:00")),
                    df.with_columns(tenji=pl.lit(float("nan"))),
                    df.with_columns(tenji=pl.lit(68.0)),
                    df.with_columns(class_ord=pl.lit(5)),
                    df.drop("tenji")]:
            with self.assertRaises((ValueError, TypeError)):
                v2.validate_input(bad)

    def test_tenji_derivation_uses_average_rank(self):
        df = v2.derive(fixture().with_columns(tenji=pl.lit(6.8)))
        self.assertEqual(df["tenji_rank"].to_list(), [3.5] * 6)
        self.assertEqual(df["tenji_rel"].to_list(), [0.0] * 6)


class ModelTest(unittest.TestCase):
    def test_second_stage_never_repeats_the_winner(self):
        rng = np.random.default_rng(0)
        rows = []
        for race in range(300):
            ranks = rng.permutation(6) + 1
            for b in range(6):
                row = {c: float(rng.random()) for c in v2.FEATURES}
                row.update(date="20260101", jcd=1 + race % 3, race_no=race, boat=b + 1, rank=int(ranks[b]))
                rows.append(row)
        df = pl.DataFrame(rows).sort(v2.BK)
        model = v2.RaceModelV2()
        model.first = v2.baseline.model([0]).set_params(max_iter=5).fit(
            model.first_matrix(model.matrix(df)), df["rank"].to_numpy().reshape(-1, 6).argmin(1))
        rank = df["rank"].to_numpy().reshape(-1, 6)
        model.second = v2.baseline.model([0, 1]).set_params(max_iter=5).fit(
            model.second_matrix(model.matrix(df), rank.argmin(1)), np.argsort(rank, 1)[:, 1])
        p1, joint = model.predict(df)
        np.testing.assert_allclose(joint.sum(axis=1), 1)
        # 1着ごとの周辺は1着確率に一致する（2着の条件付き分布が5艇で和1）
        for a in range(6):
            np.testing.assert_allclose(joint[:, v2.PAIRS[:, 0] == a].sum(axis=1), p1[:, a])
        self.assertEqual(joint.shape, (300, 30))

    def test_probability_rows_marginals(self):
        keys = fixture().select(v2.KEY).unique()
        out = v2.probability_rows(keys, np.full((1, 6), 1 / 6), np.full((1, 30), 1 / 30))
        self.assertEqual(out.height, 30)
        self.assertAlmostEqual(out["p_2t"].sum(), 1)
        self.assertAlmostEqual(out["p_2f"].sum(), 1)

    def test_training_and_prediction_features_match(self):
        """学習時（expanding）と予測時（history_as_of）で同じ特徴量になること。"""
        path = v2.ROOT / "data/parsed/results.parquet"
        if not path.exists():
            self.skipTest("local data missing")
        date = "20260905"
        train = v2.build_frame(end=date).filter(pl.col("date") == date).sort(v2.BK)
        entries = pl.read_parquet(v2.ROOT / "data/parsed/entries.parquet").filter(pl.col("date") == date).with_columns(
            class_ord=pl.col("class").replace_strict(v2.baseline.CLASS_ORD, default=None))
        tenji = pl.read_parquet(path).filter(pl.col("date") == date).select(*v2.BK, "tenji")
        live = v2.with_history(entries.join(tenji, on=v2.BK), pl.read_parquet(path)).join(
            train.select(v2.KEY).unique(), on=v2.KEY).sort(v2.BK)
        self.assertEqual(train.height, live.height)
        a = train.select(v2.FEATURES).to_numpy().astype(float)
        b = live.select(v2.FEATURES).to_numpy().astype(float)
        np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True)

    def test_saved_artifact_and_prediction_entrypoint(self):
        artifact = v2.OUT / "model.pkl"
        if not artifact.exists():
            self.skipTest("Run live_v2.py train first")
        source = pl.read_csv(v2.v1.OUT / "input_example.csv", schema_overrides={"date": pl.String})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, target = directory / "input.csv", directory / "prediction.csv"
            source.write_csv(path)
            with self.assertRaisesRegex(ValueError, "training cutoff"):
                v2.predict(path, target)
            # 日付をずらした合成入力。インターフェースの検査で、実際の予測ではない
            source = source.with_columns(date=pl.lit("20260912"), **{
                c: pl.col(c).str.replace(r"^\d{4}-\d{2}-\d{2}", "2026-09-12")
                for c in v2.v1.TIMES + ["deadline_at"]})
            source.write_csv(path)
            v2.predict(path, target)
            output = pl.read_csv(target)
            self.assertEqual(output.height, 30)
            self.assertAlmostEqual(output["p_2t"].sum(), 1)
            self.assertAlmostEqual(output["p_2f"].sum(), 1)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                v2.predict(path, target)


class MarketTest(unittest.TestCase):
    def test_quinella_from_exacta(self):
        p = np.arange(1, 31, dtype=float)[None]
        q = market.quinella_from_exacta(p)
        index = market.EXACTA_INDEX
        for j, (a, b) in enumerate(market.POOLS["2f"]):
            self.assertEqual(q[0, j], p[0, index[(a, b)]] + p[0, index[(b, a)]])
        self.assertEqual(q.sum(), p.sum())

    def test_fit_recovers_market_when_fundamental_is_useless(self):
        p = np.tile([1 / 3] * 3, (300, 1))
        q = np.tile([.7, .2, .1], (300, 1))
        y = np.repeat([0, 1, 2], [210, 60, 30])
        c = market.fit([np.log(p), np.log(q)], y)
        self.assertAlmostEqual(c[1], 1, places=3)
        np.testing.assert_allclose(market.distribution([np.log(p), np.log(q)], c).sum(axis=1), 1)

    def test_board_checksum_catches_typos(self):
        good = pl.concat([realistic_board("2t"), realistic_board("2f")])
        market.validate_odds(good)
        favorite = realistic_board("2t").with_columns(
            odds=pl.when((pl.col("first") == 1) & (pl.col("second") == 2)).then(pl.col("odds") * 10)
            .otherwise(pl.col("odds")))
        with self.assertRaisesRegex(ValueError, "transcription"):
            market.validate_odds(favorite)
        # 元返し 1.0倍がある板は Σ が下限を割っても通す
        capped = realistic_board("2f").with_columns(
            odds=pl.when((pl.col("first") == 1) & (pl.col("second") == 2)).then(1.0)
            .otherwise(pl.col("odds") * 6))
        market.validate_odds(capped)

    def test_settle_factor_neutral_and_direction(self):
        o2t = realistic_board("2t")["odds"].to_numpy()[None]
        q = market.normalize_inverse(o2t)
        design = market.drift_design("2t", q, o2t, np.full((1, 15), np.nan))
        neutral = dict(beta=[0, 0, 0], smearing=1.0)
        np.testing.assert_allclose(market.settle_factor(design, neutral), 1)
        # モデルが市場より高く見る目ほど、確定までにオッズが縮む（傾き負）
        drift = dict(beta=[0, -.7, 0], smearing=1.0)
        high = market.drift_design("2t", np.clip(q * 2, 0, 1), o2t, np.full((1, 15), np.nan))
        self.assertTrue((market.settle_factor(high, drift) < 1).all())

    def test_full_saved_pipeline(self):
        if not (market.OUT / "market_coefficients.json").exists():
            self.skipTest("Run live_market_v2.py train first")
        source = pl.read_csv(v2.v1.OUT / "input_example.csv", schema_overrides={"date": pl.String})
        source = source.with_columns(date=pl.lit("20260912"), **{
            c: pl.col(c).str.replace(r"^\d{4}-\d{2}-\d{2}", "2026-09-12")
            for c in v2.v1.TIMES + ["deadline_at"]})
        row = source.row(0, named=True)
        args = dict(jcd=row["jcd"], race_no=row["race_no"], captured=row["tkz_captured_at"],
                    deadline=row["deadline_at"])
        odds = pl.concat([realistic_board("2t", **args), realistic_board("2f", **args)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.write_csv(root / "input.csv")
            odds.write_csv(root / "odds.csv")
            market.predict(root / "input.csv", root / "odds.csv", root / "output")
            result = pl.read_csv(root / "output/final.csv")
            self.assertEqual(result.height, 45)
            for pool in market.POOLS:
                self.assertAlmostEqual(result.filter(pl.col("pool") == pool)["p_final"].sum(), 1)
            np.testing.assert_allclose(result["ev_captured"].to_numpy(),
                                       result["p_final"].to_numpy() * result["odds"].to_numpy())
            self.assertEqual(result["paper_candidate"].to_list(), (result["ev_settle"] > 1).to_list())
            with self.assertRaisesRegex(ValueError, "already exists"):
                market.predict(root / "input.csv", root / "odds.csv", root / "output")
            odds.filter(pl.col("pool") == "2f").write_csv(root / "odds_2f_only.csv")
            with self.assertRaisesRegex(ValueError, "2連単 board is required"):
                market.predict(root / "input.csv", root / "odds_2f_only.csv", root / "output2")


if __name__ == "__main__":
    unittest.main()
