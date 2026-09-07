"""第1段 fundamental モデル（オッズ非使用）→ G1 判定。

- HistGradientBoosting で艇ごとの1着確率 → レース内正規化
- walk-forward: 月単位で「その月より前の全データ」で学習し当月を予測
- G1: 素朴事前（会場×枠の過去勝率を正規化しただけの確率）を
      1着者 log-loss で有意に上回るか。有意性は レース単位ブートストラップ。

実行: uv run python src/train_fundamental.py [--min-train-days 60]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent

FEATURES = [
    "boat",  # 枠
    "jcd",
    "class_ord",
    "age",
    "weight",
    "nat_wr",
    "nat_2r",
    "loc_wr",
    "loc_2r",
    "motor_2r",
    "boat_2r",
    "tenji",
    "tenji_rank",
    "tenji_rel",
    "hist_n",
    "hist_win_rate",
    "hist_top2_rate",
    "hist_mean_st",
    "hist_f_rate",
    "vb_win_rate",
]
CATEGORICAL = ["boat", "jcd"]


def race_logloss(df: pl.DataFrame, prob_col: str) -> pl.DataFrame:
    """レースごとに -log p(勝者)。prob_col はレース内で正規化済み前提。"""
    return (
        df.filter(pl.col("y_win") == 1)
        .select(
            "date",
            "jcd",
            "race_no",
            ll=-pl.col(prob_col).clip(1e-9, 1.0).log(),
        )
    )


def normalize(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    return df.with_columns(
        (pl.col(col) / pl.col(col).sum().over(["date", "jcd", "race_no"])).alias(out)
    )


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """mean(a-b) のブートストラップ。a,b はレース対応済み。"""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.quantile(means, [0.025, 0.975])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train-days", type=int, default=60)
    args = ap.parse_args()

    feat = pl.read_parquet(ROOT / "data" / "features" / "base.parquet")
    # 完走6艇のレースのみで評価（返還絡みの착順は別途）
    ok = (
        feat.group_by(["date", "jcd", "race_no"])
        .agg(ok=pl.col("y_rank").is_not_null().all())
        .filter(pl.col("ok"))
    )
    feat = feat.join(ok.select(["date", "jcd", "race_no"]), on=["date", "jcd", "race_no"])
    feat = feat.sort(["date", "jcd", "race_no", "boat"])

    months = sorted(feat["date"].str.slice(0, 6).unique().to_list())
    dates = sorted(feat["date"].unique().to_list())

    preds = []
    for m in months:
        train = feat.filter(pl.col("date").str.slice(0, 6) < m)
        test = feat.filter(pl.col("date").str.slice(0, 6) == m)
        if train["date"].n_unique() < args.min_train_days or len(test) == 0:
            continue
        X = train.select(FEATURES).to_numpy()
        y = train["y_win"].to_numpy()
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
            random_state=0,
        )
        clf.fit(X, y)
        p = clf.predict_proba(test.select(FEATURES).to_numpy())[:, 1]
        preds.append(test.with_columns(p_raw=pl.lit(pl.Series(p))))
        print(f"{m}: train={len(train)} test={len(test)}", flush=True)

    if not preds:
        print("学習可能な月がない（データ不足）")
        return
    pred = pl.concat(preds)
    pred = normalize(pred, "p_raw", "p_fund")
    # 素朴事前: 会場×枠の過去勝率（時刻安全に構築済みの vb_win_rate）を正規化
    pred = pred.with_columns(pl.col("vb_win_rate").fill_null(1 / 6))
    pred = normalize(pred, "vb_win_rate", "p_naive")

    ll_f = race_logloss(pred, "p_fund").sort(["date", "jcd", "race_no"])
    ll_n = race_logloss(pred, "p_naive").sort(["date", "jcd", "race_no"])
    j = ll_f.join(ll_n, on=["date", "jcd", "race_no"], suffix="_naive")
    a, b = j["ll"].to_numpy(), j["ll_naive"].to_numpy()
    diff, ci = bootstrap_diff(a, b)
    print(f"\nraces={len(j)}")
    print(f"fundamental log-loss: {a.mean():.4f}")
    print(f"naive(会場×枠) log-loss: {b.mean():.4f}")
    print(f"diff: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print("G1", "PASS" if ci[1] < 0 else "FAIL", "(CI上限<0で合格)")

    out = ROOT / "data" / "preds"
    out.mkdir(parents=True, exist_ok=True)
    pred.select(
        ["date", "jcd", "race_no", "boat", "p_fund", "p_naive", "y_win", "y_rank"]
    ).write_parquet(out / "fundamental_wf.parquet")


if __name__ == "__main__":
    main()
