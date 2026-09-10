"""手動保存した実2連単板による、段階モデルの後ろ向き診断。"""

from pathlib import Path

import numpy as np
import polars as pl

from gap_at_scale import combo_index
from holdout import HOLDOUT_END

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "jcd", "race_no"]


def main() -> None:
    odds = pl.read_parquet(ROOT / "data/parsed/odds2tf_manual.parquet")
    sealed_dates = odds.filter(pl.col("date") <= HOLDOUT_END)["date"].unique()
    if sealed_dates.len():
        raise SystemExit(f"封印期間内の日付が混入している（unseal()未経由）: {sealed_dates.to_list()}")
    print("9月8日の12画像は撮影順（12R→1R→…→11R）を補正済み。")
    stage = pl.read_parquet(ROOT / "data/preds/stagewise_202609.parquet")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet")

    # 120通りから P(1着, 2着) へ周辺化する。
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
                    model_rows.append({**key, "first": first, "second": second, "p_stage_2tf": p2[row, first - 1, second - 1]})
    model = pl.DataFrame(model_rows)

    target_keys = odds.select(KEY).unique()
    ranks = (
        results.join(target_keys, on=KEY, how="inner")
        .group_by(KEY)
        .agg(
            hit_first=pl.col("boat").filter(pl.col("rank_raw") == "01").first(),
            hit_second=pl.col("boat").filter(pl.col("rank_raw") == "02").first(),
        )
    )
    pay = payouts.filter(pl.col("bet_type") == "２連単").select(KEY + ["combo", "amount"])
    df = (
        odds.join(model, on=KEY + ["first", "second"], how="inner")
        .join(ranks, on=KEY, how="inner")
        .join(pay, on=KEY + ["combo"], how="left")
        .with_columns(
            hit=((pl.col("first") == pl.col("hit_first")) & (pl.col("second") == pl.col("hit_second"))),
            q_market=(1 / pl.col("odds")) / (1 / pl.col("odds")).sum().over(KEY),
        )
        .with_columns(
            stage_rank=pl.col("p_stage_2tf").rank("ordinal", descending=True).over(KEY),
            market_rank=pl.col("q_market").rank("ordinal", descending=True).over(KEY),
        )
    )
    race = df.filter(pl.col("hit")).sort(KEY)
    if race.height == 0:
        raise SystemExit("的中着順を結果データと結合できませんでした")

    p = race["p_stage_2tf"].to_numpy()
    mismatch = race.filter(pl.col("amount").is_null() | ((pl.col("odds") * 100 - pl.col("amount")).abs() > 1))
    if mismatch.height:
        print(mismatch.select(KEY + ["combo", "odds", "amount"]))
        raise SystemExit("的中オッズと払戻に不一致。評価を停止")
    q = race["q_market"].to_numpy()
    ll_stage = -np.log(np.clip(p, 1e-12, None))
    ll_market = -np.log(np.clip(q, 1e-12, None))
    rng = np.random.default_rng(0)
    d = ll_stage - ll_market
    boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    excluded = target_keys.join(stage.select(KEY), on=KEY, how="anti").sort(KEY)
    print(f"races={race.height}  period={race['date'].min()}〜{race['date'].max()}")
    print(f"stagewise 2連単 log-loss: {ll_stage.mean():.4f}")
    print(f"market    2連単 log-loss: {ll_market.mean():.4f}")
    print(f"stagewise - market: {d.mean():+.4f}  95% bootstrap CI [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]")
    if excluded.height:
        print("モデル対象外（6艇完走条件外）:", ", ".join(f"{x[0]}-{x[1]:02d}-{x[2]:02d}R" for x in excluded.iter_rows()))

    print("\n的中2連単の順位（1位が最高確率）")
    out = ROOT / "data/diagnostics/74tukkii"
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for col, name in [("stage_rank", "model"), ("market_rank", "market")]:
        for k in (2, 3):
            bets = df.filter(pl.col(col) <= k)
            hits = bets.filter(pl.col("hit"))
            stake = bets.height * 100
            returned = hits["amount"].sum()
            summaries.append(dict(method=name, top_k=k, races=race.height, hits=hits.height, hit_rate=hits.height/race.height, stake=stake, returned=returned, roi=returned/stake))
    summary = pl.DataFrame(summaries)
    summary.write_csv(out / "top_k.csv")
    print(summary)
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").select(KEY + ["boat", "racer_id", "name"])
    for position, label in [("first", "first"), ("second", "second")]:
        marginal = df.group_by(KEY + [position]).agg(p=pl.col("p_stage_2tf").sum(), q=pl.col("q_market").sum(), actual=pl.col("hit").any().cast(pl.Int64))
        marginal = marginal.rename({position: "boat"}).join(entries, on=KEY + ["boat"], validate="1:1")
        racers = marginal.group_by("racer_id", "name").agg(n=pl.len(), actual=pl.col("actual").sum(), model_expected=pl.col("p").sum(), market_expected=pl.col("q").sum(), model_brier=((pl.col("p")-pl.col("actual"))**2).mean(), market_brier=((pl.col("q")-pl.col("actual"))**2).mean()).with_columns(gap=pl.col("model_brier")-pl.col("market_brier")).sort("gap", descending=True)
        racers.write_csv(out / f"racers_{label}.csv")
        print(label, racers.head(6))
    df.write_csv(out / "tickets.csv")
    for col, name in [("stage_rank", "stagewise"), ("market_rank", "market")]:
        ranks = race[col]
        print(
            f"{name:9}: 平均={ranks.mean():.2f}位 中央値={ranks.median():.0f}位 "
            f"top1={(ranks == 1).mean():.1%} top3={(ranks <= 3).mean():.1%} top5={(ranks <= 5).mean():.1%}"
        )

    # 事後に閾値を選ばず、診断用にあらかじめ3段階を並べる。各1点100円相当。
    for threshold in (1.00, 1.05, 1.10):
        bets = df.filter(pl.col("p_stage_2tf") * pl.col("odds") > threshold)
        returns = bets.with_columns(
            pl.when(pl.col("hit")).then(pl.col("amount").fill_null(0)).otherwise(0).alias("return_yen")
        )
        stake = 100 * returns.height
        ret = returns["return_yen"].sum()
        print(f"EV>{threshold:.2f}: bets={returns.height} stake={stake} return={ret} ROI={ret/stake:.2%}" if stake else f"EV>{threshold:.2f}: bets=0")


if __name__ == "__main__":
    main()
