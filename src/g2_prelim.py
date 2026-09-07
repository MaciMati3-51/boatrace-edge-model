"""G2 予備判定 — 手持ち34日分のオッズだけで「優位の有無」の符号を先に見る。

正式な G2（combine_market.py）は月単位 walk-forward だが、オッズが
20230701〜0803 の34日しかないため walk-forward だと検定に使える月が
202308（3日分）しか残らず検出力が出ない。予備判定として設計を分ける:

  fundamental : 2022-07-01〜2023-06-30 で1本学習 → 202307-08 を予測
                （**厳密に時系列 OOS**。ここは絶対に緩めない）
  合成器      : レース単位 K-fold CV（全レースを評価に使う）

合成器の CV は時系列を混ぜるが、合成器のパラメータは2個しかないため個別レースの
記憶は起きない。一方 fundamental 側で時系列を緩めると、同じ選手の将来の調子を
学習してしまい **G2 が偽陽性になる**（＝規約リスクを取って追加収集する判断を誤らせる）。
よって fundamental は時系列 OOS 厳守、合成器のみ CV とする。

## 合成器は条件付きロジットで推定する（sklearn の LogisticRegression ではない）

評価はレース内 softmax（6艇から1艇が勝つ条件付き多項分布）なので、推定も同じ
目的関数でなければならない。`LogisticRegression` は艇ごとの**二値**分類を最適化する
別物で、これを使うと softmax 評価上は歪んだ係数が選ばれ、**市場単体より悪い合成が
出てくる**（実測: b=1.381 が選ばれ log-loss が市場 1.1686 → 合成 1.2167 と悪化）。

条件付きロジットなら a=0, b=1 で市場を厳密に再現できるため、合成が市場を下回ることは
原理的に起きない。「市場に情報を足せるか」だけを純粋に測れる。

## 比較する3つのモデル

  1. market       q そのもの（= b を1に固定）
  2. market^b     b·log q のみ（b自由）… favorite-longshot bias の power 補正
  3. combined     a·log p_fund + b·log q … G2 の対象

G2 は 3 vs 1。3 vs 2 は「power補正を差し引いた fundamental の正味の寄与」を見る。

実行: uv run python src/g2_prelim.py
"""

import argparse

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from train_fundamental import CATEGORICAL, FEATURES, ROOT, normalize

TRAIN_START, TRAIN_END = "20220701", "20230630"
TEST_START, TEST_END = "20230701", "20230803"
KEY = ["date", "jcd", "race_no"]
N_FOLDS = 5
N_BOATS = 6


def fit_fundamental(feat: pl.DataFrame) -> pl.DataFrame:
    """2022-07〜2023-06 で学習し、202307-08 を予測（時系列 OOS）。"""
    train = feat.filter((pl.col("date") >= TRAIN_START) & (pl.col("date") <= TRAIN_END))
    test = feat.filter((pl.col("date") >= TEST_START) & (pl.col("date") <= TEST_END))
    print(
        f"fundamental 学習: {train['date'].min()}〜{train['date'].max()} "
        f"{train.select(KEY).unique().height:,}レース / "
        f"予測: {test.select(KEY).unique().height:,}レース"
    )
    if train.select(KEY).unique().height < 1000 or test.height == 0:
        raise SystemExit("学習データ不足。LZH の遡り取得が完了しているか確認せよ")

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        random_state=0,
    )
    clf.fit(train.select(FEATURES).to_numpy(), train["y_win"].to_numpy())
    p = clf.predict_proba(test.select(FEATURES).to_numpy())[:, 1]
    return normalize(test.with_columns(p_raw=pl.lit(pl.Series(p))), "p_raw", "p_fund")


def market_prob(odds: pl.DataFrame) -> pl.DataFrame:
    """3連単120通り → 1着艇へ周辺化した市場含意勝率 q。

    null（欠場等）を含むレースはプールが歪むので丸ごと落とす。
    """
    bad = (
        odds.group_by(KEY)
        .agg(nn=pl.col("odds").is_null().sum(), n=pl.len())
        .filter((pl.col("nn") > 0) | (pl.col("n") != 120))
        .select(KEY)
    )
    odds = odds.join(bad, on=KEY, how="anti").filter(pl.col("odds") > 0)
    return (
        odds.with_columns(
            inv=1 / pl.col("odds"),
            boat=pl.col("combo").str.slice(0, 1).cast(pl.Int64),
        )
        .group_by(KEY + ["boat"])
        .agg(m=pl.col("inv").sum())
        .with_columns(q=pl.col("m") / pl.col("m").sum().over(KEY))
    )


def nll(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """レース内 softmax の負の対数尤度。X:(R,6,k) y:(R,) 勝者の艇インデックス。"""
    z = X @ w
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()


def fit_clogit(X: np.ndarray, y: np.ndarray, w0: np.ndarray) -> np.ndarray:
    return minimize(nll, w0, args=(X, y), method="L-BFGS-B").x


def race_ll(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """レースごとの -log p(勝者)。"""
    z = X @ w
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    return d.mean(), np.quantile(d[idx].mean(axis=1), [0.025, 0.975])


def verdict(ll_a: np.ndarray, ll_b: np.ndarray, name_a: str, name_b: str) -> None:
    diff, ci = boot_ci(ll_a - ll_b)
    mark = "PASS" if ci[1] < 0 else "FAIL"
    print(
        f"  {name_a} - {name_b}: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  → {mark}"
    )


def load_wf_fundamental() -> pl.DataFrame:
    """本番の walk-forward 予測（fundamental_wf.parquet）から検定期間を切り出す。

    単一窓1年学習より強い fundamental で同じ検定をやり直すための経路。
    こちらも当然 OOS（train_fundamental.py が月単位 walk-forward で吐いたもの）。
    """
    wf = pl.read_parquet(ROOT / "data" / "preds" / "fundamental_wf.parquet")
    sub = wf.filter((pl.col("date") >= TEST_START) & (pl.col("date") <= TEST_END))
    print(
        f"fundamental: 本番 walk-forward 予測を使用 "
        f"({sub.select(KEY).unique().height:,}レース)"
    )
    if sub.height == 0:
        raise SystemExit(
            "fundamental_wf.parquet が検定期間を含まない。"
            "train_fundamental.py を再実行して予測範囲を確認せよ"
        )
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-wf",
        action="store_true",
        help="単一窓学習ではなく本番 walk-forward 予測を使う（検出力そのまま・より強い fundamental）",
    )
    args = ap.parse_args()

    feat = pl.read_parquet(ROOT / "data" / "features" / "base.parquet")
    ok = (
        feat.group_by(KEY)
        .agg(ok=pl.col("y_rank").is_not_null().all())
        .filter(pl.col("ok"))
        .select(KEY)
    )
    feat = feat.join(ok, on=KEY).sort(KEY + ["boat"])

    fund = load_wf_fundamental() if args.use_wf else fit_fundamental(feat)
    mkt = market_prob(pl.read_parquet(ROOT / "data" / "parsed" / "odds3t.parquet"))

    df = fund.join(mkt.select(KEY + ["boat", "q"]), on=KEY + ["boat"], how="inner")
    full = df.group_by(KEY).len().filter(pl.col("len") == N_BOATS).select(KEY)
    df = df.join(full, on=KEY).sort(KEY + ["boat"])
    # 勝者がちょうど1艇いるレースのみ
    one = df.group_by(KEY).agg(w=pl.col("y_win").sum()).filter(pl.col("w") == 1)
    df = df.join(one.select(KEY), on=KEY).sort(KEY + ["boat"])

    n_races = df.height // N_BOATS
    print(f"オッズと突合できたレース: {n_races:,}")
    if n_races < 500:
        raise SystemExit("突合レースが少なすぎる。日付範囲の重なりを確認せよ")

    lf = np.log(np.clip(df["p_fund"].to_numpy(), 1e-9, None)).reshape(n_races, N_BOATS)
    lq = np.log(np.clip(df["q"].to_numpy(), 1e-9, None)).reshape(n_races, N_BOATS)
    y = df["y_win"].to_numpy().reshape(n_races, N_BOATS).argmax(axis=1)

    X_comb = np.stack([lf, lq], axis=2)   # (R,6,2)
    X_mkt = lq[:, :, None]                # (R,6,1)

    # モデル1: 市場そのもの（b=1固定）
    ll_market = race_ll(X_mkt, y, np.array([1.0]))
    ll_fund = race_ll(lf[:, :, None], y, np.array([1.0]))

    # モデル2,3: レース単位 K-fold CV
    fold = np.arange(n_races) % N_FOLDS
    ll_pow = np.empty(n_races)
    ll_comb = np.empty(n_races)
    coefs = []
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        w_pow = fit_clogit(X_mkt[tr], y[tr], np.array([1.0]))
        w_comb = fit_clogit(X_comb[tr], y[tr], np.array([0.0, 1.0]))
        ll_pow[te] = race_ll(X_mkt[te], y[te], w_pow)
        ll_comb[te] = race_ll(X_comb[te], y[te], w_comb)
        coefs.append((w_pow[0], w_comb[0], w_comb[1]))

    bp, a, b = (float(np.mean([c[i] for c in coefs])) for i in range(3))
    print(f"\n係数（{N_FOLDS}-fold平均・条件付きロジット）")
    print(f"  market^b      : b={bp:.3f}   (b>1 は人気薄の過大評価=favorite-longshot bias の補正)")
    print(f"  combined      : a(fundamental)={a:.3f}  b(market)={b:.3f}")

    print(f"\n===== log-loss  races={n_races:,} =====")
    print(f"  market   (b=1固定) : {ll_market.mean():.4f}")
    print(f"  market^b (power補正): {ll_pow.mean():.4f}")
    print(f"  fundamental 単体   : {ll_fund.mean():.4f}")
    print(f"  combined           : {ll_comb.mean():.4f}")

    print("\n===== 判定（CI上限<0で合格）=====")
    verdict(ll_comb, ll_market, "combined", "market  ")
    print("    ↑ これが G2 本体")
    verdict(ll_comb, ll_pow, "combined", "market^b")
    print("    ↑ power補正を差し引いた fundamental の正味の寄与")
    verdict(ll_pow, ll_market, "market^b", "market  ")
    print("    ↑ 参考: power補正だけでどれだけ改善するか")


if __name__ == "__main__":
    main()
