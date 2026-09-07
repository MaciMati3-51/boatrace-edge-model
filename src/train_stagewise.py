"""段階条件付き fundamental モデルを月次 walk-forward で学習する。

既存の ``fundamental_wf.parquet`` にある1着確率は固定し、次の2モデルを追加する。

    P(1着=a) × P(2着=b | 1着=a) × P(3着=c | 1着=a, 2着=b)

2着モデルは実際の1着艇、3着モデルは実際の1・2着艇を条件として学習する。
予測時は全ての仮想的な (a,b,c) を列挙するので、結果情報は特徴量に入らない。
各段の候補スコアは残っている艇の中で正規化する。

出力は1レース1行。``p_stage`` は combo_index() と同じ順序の120要素である。
封印側も因果的な予測だけは生成するが、集計・表示は walk-forward 側に限定する。

実行: uv run python src/train_stagewise.py
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from gap_at_scale import combo_index, paired_ci, pl_prob
from holdout import HOLDOUT_START, walk_forward
from train_fundamental import CATEGORICAL, FEATURES, ROOT

KEY = ["date", "jcd", "race_no"]
N_BOATS = 6

# 先着艇の状態として渡す列。艇番だけでなく能力・展示値も渡すことで、
# 「誰が先着したか」によって残り艇の着順が変わる相互作用を木が拾える。
CONTEXT_FEATURES = [
    "boat",
    "class_ord",
    "nat_wr",
    "nat_2r",
    "loc_wr",
    "loc_2r",
    "motor_2r",
    "boat_2r",
    "tenji",
    "tenji_rank",
    "tenji_rel",
    "hist_win_rate",
    "hist_top2_rate",
    "hist_mean_st",
    "hist_f_rate",
    "vb_win_rate",
]
CONTEXT_IDX = np.array([FEATURES.index(c) for c in CONTEXT_FEATURES])
DELTA_IDX = np.array(
    [FEATURES.index(c) for c in CONTEXT_FEATURES if c != "boat"]
)


def model(categorical_features: list[int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        categorical_features=categorical_features,
        random_state=0,
    )


def conditional_features(
    x: np.ndarray, candidate: np.ndarray, previous: list[np.ndarray]
) -> np.ndarray:
    """候補艇と先着艇から、stage 2/3 共通の条件付き特徴量を作る。

    x は (race, boat, feature)。candidate / previous は各行の艇index。
    候補自身の特徴、先着艇の状態、候補との差を順に連結する。
    """
    rows = np.arange(len(candidate))
    candidate_x = x[rows, candidate]
    parts = [candidate_x]
    for prev in previous:
        prev_x = x[rows, prev]
        parts.extend(
            [prev_x[:, CONTEXT_IDX], candidate_x[:, DELTA_IDX] - prev_x[:, DELTA_IDX]]
        )
    return np.column_stack(parts)


def categorical_indices(stage: int) -> list[int]:
    """conditional_features のうちカテゴリとして扱う列index。"""
    out = [FEATURES.index(c) for c in CATEGORICAL]
    block = len(CONTEXT_FEATURES) + len(DELTA_IDX)
    first_context_boat = len(FEATURES)
    for i in range(stage - 1):
        out.append(first_context_boat + i * block)  # previous boat
    return out


def training_rows(x: np.ndarray, rank: np.ndarray, stage: int) -> tuple[np.ndarray, np.ndarray]:
    """実際の先着順を条件に、当該stageで残っていた候補行を作る。"""
    order = np.argsort(rank, axis=1)
    keep = rank >= stage
    race, candidate = np.nonzero(keep)
    previous = [order[race, i] for i in range(stage - 1)]
    return conditional_features(x[race], candidate, previous), (rank[race, candidate] == stage)


def conditional_scores(clf, x: np.ndarray, stage: int, tri: np.ndarray) -> np.ndarray:
    """全レース・全組について当該stage候補の生スコアを返す。"""
    n, k = len(x), len(tri)
    candidate = np.tile(tri[:, stage - 1], n)
    previous = [np.tile(tri[:, i], n) for i in range(stage - 1)]
    race = np.repeat(np.arange(n), k)
    z = conditional_features(x[race], candidate, previous)
    return clf.predict_proba(z)[:, 1].reshape(n, k)


def stage_probabilities(
    first: np.ndarray, score2: np.ndarray, score3: np.ndarray, tri: np.ndarray
) -> np.ndarray:
    """段ごとのスコアを条件付き正規化し、120通りの同時確率にする。"""
    n = len(first)
    p2 = np.empty_like(score2)
    p3 = np.empty_like(score3)

    for a in range(N_BOATS):
        m2 = tri[:, 0] == a
        den2 = score2[:, m2].sum(axis=1, keepdims=True)
        p2[:, m2] = score2[:, m2] / np.clip(den2, 1e-12, None)
        for b in range(N_BOATS):
            if a == b:
                continue
            m3 = m2 & (tri[:, 1] == b)
            den3 = score3[:, m3].sum(axis=1, keepdims=True)
            p3[:, m3] = score3[:, m3] / np.clip(den3, 1e-12, None)

    p = first[:, tri[:, 0]] * p2 * p3
    return p / p.sum(axis=1, keepdims=True)


def array_series(name: str, values: np.ndarray) -> pl.Series:
    return pl.Series(name, values, dtype=pl.Array(pl.Float32, values.shape[1]))


def evaluate(pred: pl.DataFrame, fundamental: pl.DataFrame, tri: np.ndarray) -> None:
    """封印前だけで PL と段階モデルの120通りlog-lossを比較する。"""
    market = walk_forward(
        pl.read_parquet(ROOT / "data" / "parsed" / "market_payout.parquet")
    )
    ev = (
        walk_forward(pred)
        .join(market.select(KEY + ["combo", "q_3t"]), on=KEY)
        .sort(KEY)
    )
    if ev.is_empty():
        print("評価対象がない")
        return

    pos = {f"{a + 1}-{b + 1}-{c + 1}": i for i, (a, b, c) in enumerate(tri)}
    hit = np.array([pos[c] for c in ev["combo"].to_list()])
    p_stage = ev["p_stage"].to_numpy()
    p_hit = p_stage[np.arange(len(ev)), hit]

    f = (
        fundamental.join(ev.select(KEY), on=KEY)
        .sort(KEY + ["boat"])
    )
    s = f["p_fund"].to_numpy().reshape(len(ev), N_BOATS)
    p_pl = pl_prob(s, tri)[np.arange(len(ev)), hit]
    q = ev["q_3t"].to_numpy()

    ll_stage = -np.log(np.clip(p_hit, 1e-12, None))
    ll_pl = -np.log(np.clip(p_pl, 1e-12, None))
    ll_market = -np.log(np.clip(q, 1e-12, None))

    print(f"\n===== 封印前 120通り log-loss  races={len(ev):,} =====")
    print(f"  市場                : {ll_market.mean():.4f}")
    print(f"  fundamental (PL)    : {ll_pl.mean():.4f}")
    print(f"  fundamental (段階)  : {ll_stage.mean():.4f}")
    for a, b, name in [
        (ll_stage, ll_pl, "段階 - PL"),
        (ll_stage, ll_market, "段階 - 市場"),
    ]:
        diff, ci, bci = paired_ci(a - b)
        print(
            f"  {name:<18}: {diff:+.4f}  "
            f"95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
            f"bootstrap [{bci[0]:+.4f}, {bci[1]:+.4f}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train-days", type=int, default=60)
    ap.add_argument("--start-month", default="", help="動作確認用。予測開始月 YYYYMM")
    ap.add_argument("--end-month", default="", help="動作確認用。予測終了月 YYYYMM")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "data" / "preds" / "stagewise_wf.parquet"
    )
    args = ap.parse_args()

    feat = pl.read_parquet(ROOT / "data" / "features" / "base.parquet")
    ok = (
        feat.group_by(KEY)
        .agg(n=pl.len(), complete=pl.col("y_rank").is_not_null().all())
        .filter((pl.col("n") == N_BOATS) & pl.col("complete"))
        .select(KEY)
    )
    feat = feat.join(ok, on=KEY).sort(KEY + ["boat"])
    fundamental = pl.read_parquet(ROOT / "data" / "preds" / "fundamental_wf.parquet")

    months = sorted(fundamental["date"].str.slice(0, 6).unique().to_list())
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]

    tri = combo_index()
    predictions = []
    for month in months:
        train = feat.filter(pl.col("date").str.slice(0, 6) < month)
        test_keys = (
            fundamental.filter(pl.col("date").str.slice(0, 6) == month)
            .select(KEY)
            .unique()
        )
        test = feat.join(test_keys, on=KEY).sort(KEY + ["boat"])
        if train["date"].n_unique() < args.min_train_days or test.is_empty():
            continue

        n_train, n_test = train.height // N_BOATS, test.height // N_BOATS
        x_train = train.select(FEATURES).to_numpy().reshape(n_train, N_BOATS, -1)
        rank_train = train["y_rank"].to_numpy().reshape(n_train, N_BOATS)
        x_test = test.select(FEATURES).to_numpy().reshape(n_test, N_BOATS, -1)

        x2, y2 = training_rows(x_train, rank_train, 2)
        x3, y3 = training_rows(x_train, rank_train, 3)
        clf2, clf3 = model(categorical_indices(2)), model(categorical_indices(3))
        clf2.fit(x2, y2)
        clf3.fit(x3, y3)

        score2 = conditional_scores(clf2, x_test, 2, tri)
        score3 = conditional_scores(clf3, x_test, 3, tri)
        first_rows = (
            fundamental.join(test_keys, on=KEY)
            .sort(KEY + ["boat"])
        )
        first = first_rows["p_fund"].to_numpy().reshape(n_test, N_BOATS)
        p = stage_probabilities(first, score2, score3, tri).astype(np.float32)

        races = test.select(KEY).unique(maintain_order=True)
        predictions.append(races.with_columns(array_series("p_stage", p)))
        print(
            f"{month}: train={n_train:,} races test={n_test:,} races "
            f"p_sum=[{p.sum(axis=1).min():.6f}, {p.sum(axis=1).max():.6f}]",
            flush=True,
        )

    if not predictions:
        raise SystemExit("学習可能な月がない")
    pred = pl.concat(predictions).sort(KEY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(args.output)
    print(
        f"\n保存: {args.output}  {pred.height:,}レース "
        f"({pred['date'].min()}〜{pred['date'].max()})"
    )
    if args.start_month or args.end_month:
        print("期間指定の動作確認なので評価を省略")
    else:
        print(f"評価は {HOLDOUT_START} より前だけ。封印側の集計は行わない")
        evaluate(pred, fundamental, tri)


if __name__ == "__main__":
    main()
