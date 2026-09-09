"""第1段 fundamental モデル（オッズ非使用）→ G1 判定。

**レース単位の6クラス softmax で解く。** 6艇ぶんの特徴量を1行にまとめ、
「どの艇が1着か」を6クラス分類する。出力は最初からレース内で和が1になる。

- walk-forward: 月単位で「その月より前の全データ」で学習し当月を予測
- G1: 素朴事前（会場×枠の過去勝率を正規化しただけの確率）を
      1着者 log-loss で有意に上回るか。有意性は レース単位ブートストラップ。

## なぜ「艇ごと二値分類 → レース内で正規化」をやめたか（2026-09-09）

旧実装は艇ごとに `y_win` を二値分類し、後からレース内で正規化していた。特徴量は
自艇の属性だけなので、**モデルは相手5艇を1つも見ていない**。出せるのは「自分の属性の
周辺勝率」だけで、正規化が誤差を再配分して本命の spread を潰す。

`diagnose_first_stage.py` の実測（walk-forward 147,922レース、市場も板も不要）:

    枠1 × A1   実 71.19%  →  旧モデル 59.22%   (-11.97pt)
    枠1 × B2   実 34.85%  →  旧モデル 46.31%   (+11.46pt)
    A1-B2 スプレッド  実測 36.3pt / 旧モデル 12.9pt = 実物の 36%

枠1の平均予測 55.03% に対し実勝率 55.25% で**平均は合っている**。つまり calibration
ではなく **resolution の欠損**で、温度(power)補正では直らない（t=1.075 で改善ゼロ）。
枠2〜6のバイアスは±1〜2ptと小さく、損失は確率質量の54%を持つ枠1に集中していた。

レース単位 softmax に替えると 45,272レースで 1.2138 → 1.1851（-0.0287 ±0.0027）、
板4,901レースでは市場との差 +0.0850 → +0.0624（27%縮小）。改善の1/3は単調再較正で
届く分、2/3は「相手を見た」ことによる新情報。

実行: uv run python src/train_fundamental.py [--min-train-days 60]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from holdout import HOLDOUT_START, walk_forward

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "jcd", "race_no"]
N_BOATS = 6

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

# レース単位の列構成: [jcd] + 艇1..6 の特徴量。枠は列の位置自体が表すので落とし、
# jcd はレース共通なので6回並べず先頭に1つだけ置く（唯一のカテゴリ列）。
PER_BOAT = [c for c in FEATURES if c not in ("boat", "jcd")]


def race_logloss(df: pl.DataFrame, prob_col: str) -> pl.DataFrame:
    """レースごとに -log p(勝者)。prob_col はレース内で正規化済み前提。"""
    return (
        df.filter(pl.col("y_win") == 1)
        .select(*KEY, ll=-pl.col(prob_col).clip(1e-9, 1.0).log())
    )


def normalize(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    return df.with_columns((pl.col(col) / pl.col(col).sum().over(KEY)).alias(out))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """mean(a-b) のブートストラップ。a,b はレース対応済み。"""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    return d.mean(), np.quantile(d[idx].mean(axis=1), [0.025, 0.975])


def race_matrix(feat: pl.DataFrame) -> np.ndarray:
    """(レース, 6, 特徴量) → (レース, 1 + 6×特徴量) の平坦化。先頭列は jcd。"""
    n = feat.height // N_BOATS
    jcd = feat["jcd"].to_numpy().astype(float).reshape(n, N_BOATS)[:, :1]
    x = feat.select(PER_BOAT).to_numpy().astype(float).reshape(n, -1)
    return np.hstack([jcd, x])


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        categorical_features=[0],  # jcd
        random_state=0,
    )


def bias_table(pred: pl.DataFrame) -> None:
    """枠×級別の予測バイアス。旧実装の欠陥がここに出ていたので毎回見る。"""
    t = (
        walk_forward(pred)
        .group_by(["boat", "class"])
        .agg(k=pl.len(), actual=pl.col("y_win").mean(), p=pl.col("p_fund").mean())
        .filter(pl.col("k") >= 300)
        .sort(["boat", "class"])
    )
    print("\n枠×級別の予測バイアス（walk-forward、|差|が大きいほど resolution 不足）")
    print("  枠 級別       n      実勝率    モデル      差")
    for r in t.iter_rows(named=True):
        print(f"  {r['boat']}  {r['class']:3}  {r['k']:7,}  {r['actual']:7.2%}  "
              f"{r['p']:7.2%}  {r['p'] - r['actual']:+7.2%}")
    a = t.filter((pl.col("boat") == 1) & (pl.col("class") == "A1")).row(0, named=True)
    z = t.filter((pl.col("boat") == 1) & (pl.col("class") == "B2")).row(0, named=True)
    print(f"  枠1の A1-B2 スプレッド: 実測 {a['actual'] - z['actual']:+.1%} / "
          f"モデル {a['p'] - z['p']:+.1%}  → 実物の "
          f"{(a['p'] - z['p']) / (a['actual'] - z['actual']):.0%}（旧実装は 36%）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train-days", type=int, default=60)
    ap.add_argument("--output", type=Path, default=ROOT / "data/preds/fundamental_wf.parquet")
    args = ap.parse_args()

    feat = pl.read_parquet(ROOT / "data/features/base.parquet")
    # 完走6艇のレースのみ（返還絡みの着順は別途）。同着1着は学習から外すが予測はする。
    ok = (
        feat.group_by(KEY)
        .agg(n=pl.len(), complete=pl.col("y_rank").is_not_null().all(), w=pl.col("y_win").sum())
        .filter((pl.col("n") == N_BOATS) & pl.col("complete"))
    )
    feat = feat.join(ok.select(KEY), on=KEY).sort(KEY + ["boat"])
    races = feat.select(KEY).unique(maintain_order=True)
    n = races.height
    print(f"対象 {n:,}レース（{races['date'].min()}〜{races['date'].max()}）")

    X = race_matrix(feat)
    win = feat["y_win"].to_numpy().astype(np.int8).reshape(n, N_BOATS)
    y = win.argmax(1)
    trainable = win.sum(1) == 1  # 1着同着（3レース）は学習から外す。予測はする
    dates = races["date"].to_numpy()
    month = races["date"].str.slice(0, 6).to_numpy()

    p_fund = np.full((n, N_BOATS), np.nan)
    predicted = np.zeros(n, dtype=bool)
    for m in sorted(set(month)):
        te = month == m
        tr = (month < m) & trainable
        if len(set(dates[tr])) < args.min_train_days or not te.any():
            continue
        clf = model().fit(X[tr], y[tr])
        p_fund[te] = clf.predict_proba(X[te])
        predicted[te] = True
        print(f"{m}: train={tr.sum():,}R test={te.sum():,}R", flush=True)

    if not predicted.any():
        raise SystemExit("学習可能な月がない（データ不足）")

    pred = (
        feat.with_columns(p_fund=pl.Series(p_fund.reshape(-1)))
        .with_columns(pl.col("vb_win_rate").fill_null(1 / N_BOATS))
        .pipe(normalize, "vb_win_rate", "p_naive")
        .filter(pl.col("p_fund").is_not_nan())
    )

    # 予測は全期間ぶん残す（開封時に必要）。**評価だけ**を walk-forward 側に限る。
    ev = walk_forward(pred)
    print(f"\nG1 評価は封印側を除外: {HOLDOUT_START} より前の {ev['date'].n_unique()}日")
    j = (
        race_logloss(ev, "p_fund").sort(KEY)
        .join(race_logloss(ev, "p_naive").sort(KEY), on=KEY, suffix="_naive")
    )
    a, b = j["ll"].to_numpy(), j["ll_naive"].to_numpy()
    diff, ci = bootstrap_diff(a, b)
    print(f"\nraces={j.height:,}")
    print(f"fundamental log-loss: {a.mean():.4f}")
    print(f"naive(会場×枠) log-loss: {b.mean():.4f}")
    print(f"diff: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print("G1", "PASS" if ci[1] < 0 else "FAIL", "(CI上限<0で合格)")
    bias_table(pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.select(KEY + ["boat", "p_fund", "p_naive", "y_win", "y_rank"]).write_parquet(args.output)
    print(f"\n保存: {args.output}  {pred.select(KEY).unique().height:,}レース")


if __name__ == "__main__":
    main()
