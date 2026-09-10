"""手動2連単板で、段階モデルが市場に劣後する条件を分解する診断。"""

from pathlib import Path

import numpy as np
import polars as pl

from gap_at_scale import combo_index
from holdout import HOLDOUT_END


ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "jcd", "race_no"]


def band(expr: pl.Expr, breaks: list[float], labels: list[str]) -> pl.Expr:
    out = pl.lit(labels[-1])
    for edge, label in reversed(list(zip(breaks, labels[:-1]))):
        out = pl.when(expr < edge).then(pl.lit(label)).otherwise(out)
    return out


def report(df: pl.DataFrame, column: str) -> None:
    summary = (
        df.group_by(column)
        .agg(
            n=pl.len(),
            stage_ll=pl.col("ll_stage").mean(),
            market_ll=pl.col("ll_market").mean(),
            diff=pl.col("ll_diff").mean(),
        )
        .filter(pl.col("n") >= 5)
        .sort("diff", descending=True)
    )
    print(f"\n--- {column}（diff>0 はモデル劣後、n>=5） ---")
    for row in summary.iter_rows(named=True):
        print(f"{str(row[column]):>12}  n={row['n']:2d}  stage={row['stage_ll']:.3f}  market={row['market_ll']:.3f}  diff={row['diff']:+.3f}")


def main() -> None:
    odds = pl.read_parquet(ROOT / "data/parsed/odds2tf_manual.parquet")
    sealed_dates = odds.filter(pl.col("date") <= HOLDOUT_END)["date"].unique()
    if sealed_dates.len():
        raise SystemExit(f"封印期間内の日付が混入している（unseal()未経由）: {sealed_dates.to_list()}")
    stage = pl.read_parquet(ROOT / "data/preds/stagewise_202609.parquet")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    features = pl.read_parquet(ROOT / "data/features/base.parquet")

    tri = combo_index()
    p120 = stage["p_stage"].to_numpy()
    p2 = np.zeros((len(stage), 6, 6), dtype=np.float64)
    for i, (a, b, _) in enumerate(tri):
        p2[:, a, b] += p120[:, i]
    model_rows = []
    for row, key in enumerate(stage.select(KEY).iter_rows(named=True)):
        for first in range(1, 7):
            for second in range(1, 7):
                if first != second:
                    model_rows.append({**key, "first": first, "second": second, "p_stage": p2[row, first - 1, second - 1]})
    model = pl.DataFrame(model_rows)

    target = odds.select(KEY).unique()
    ranks = (
        results.join(target, on=KEY, how="inner")
        .group_by(KEY)
        .agg(
            hit_first=pl.col("boat").filter(pl.col("rank_raw") == "01").first(),
            hit_second=pl.col("boat").filter(pl.col("rank_raw") == "02").first(),
            wind_speed=pl.col("wind_speed").first(),
        )
    )
    winner_feature = features.select(KEY + ["boat", "tenji_rank", "tenji_rel"]).rename({"boat": "hit_first"})
    all_rows = (
        odds.join(model, on=KEY + ["first", "second"], how="inner")
        .join(ranks, on=KEY, how="inner")
        .with_columns(q=(1 / pl.col("odds")) / (1 / pl.col("odds")).sum().over(KEY))
        .with_columns(
            market_rank=pl.col("q").rank("ordinal", descending=True).over(KEY),
            model_rank=pl.col("p_stage").rank("ordinal", descending=True).over(KEY),
            market_fav_q=pl.col("q").max().over(KEY),
        )
    )
    hit = (
        all_rows.filter((pl.col("first") == pl.col("hit_first")) & (pl.col("second") == pl.col("hit_second")))
        .join(winner_feature, on=KEY + ["hit_first"], how="left")
        .with_columns(
            ll_stage=-pl.col("p_stage").clip(1e-12, 1).log(),
            ll_market=-pl.col("q").clip(1e-12, 1).log(),
        )
        .with_columns(
            ll_diff=pl.col("ll_stage") - pl.col("ll_market"),
            winner_odds_band=band(pl.col("odds"), [5, 10, 20, 50], ["<5", "5-10", "10-20", "20-50", "50+"]),
            market_rank_band=band(pl.col("market_rank"), [2, 4, 7], ["1", "2-3", "4-6", "7+"]),
            fav_q_band=band(pl.col("market_fav_q"), [.20, .30, .40], ["<.20", ".20-.30", ".30-.40", ".40+"]),
            wind_band=band(pl.col("wind_speed").fill_null(-1), [0, 3, 6], ["unknown", "0-2m", "3-5m", "6m+"]),
            tenji_band=pl.col("tenji_rank").cast(pl.Int64).cast(pl.String),
        )
        .sort(KEY)
    )
    print(f"対象 {hit.height}R。全体 diff={hit['ll_diff'].mean():+.3f}（正=市場より悪い）")
    for col in ["hit_first", "hit_second", "market_rank_band", "winner_odds_band", "fav_q_band", "wind_band", "tenji_band", "race_no"]:
        report(hit, col)


if __name__ == "__main__":
    main()
