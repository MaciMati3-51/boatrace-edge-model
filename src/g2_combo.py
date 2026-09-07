"""G2 を「3連単120通り空間」で回す。既存の G2 は6艇空間（1着だけ）しかやっていない。

2026-09-07 の分解で、市場との差 +0.3888 のうち **78%（+0.3039）が「1着を所与とした
2・3着」にある**と分かった（`gap_at_scale.py --decompose`）。ところが
`g2_prelim.py` / `combine_market.py` の G2 は6艇空間、つまり残り22%の土俵でしか
合成を試していない。**120通り空間の合成は一度も走っていない。**

このスクリプトはそこを埋める。新しいデータは要らない。板のある 20230701〜0803 の
5,000レース強を使う（**全て walk-forward 側。封印には触れない**）。

## 本当の目的は検出力の値付け

板が5,000レースしかないので、a がゼロと出ても「効果が無い」のか「見えないだけ」なのか
決まらない。**知りたいのは a の値より CI の幅**。それが分かって初めて
「板を9,800円で買い足す価値があるか」を数字で判断できる（README §13 の入手経路）。

6艇空間の G2 と同じレースで両方回して並べる。空間を変えたことで検出力が
どう変わるかが見える。

## 合成器は条件付きロジット（120肢版）

    z_c = a·log p_c + b·log q_c   → 120肢の softmax → 的中目の -log P

a=0,b=1 で市場を厳密再現するので、合成が市場を下回ることは原理的に起きない
（理由は g2_prelim.py 冒頭。sklearn の LogisticRegression を使ってはいけない）。

実行: uv run --with scipy python src/g2_combo.py
"""

import argparse

import numpy as np
import polars as pl

from g2_prelim import fit_clogit, market_prob, race_ll
from holdout import HOLDOUT_START, walk_forward
from train_fundamental import ROOT

KEY = ["date", "jcd", "race_no"]
N_BOATS = 6
N_COMBO = 120
N_FOLDS = 5


def combo_list() -> tuple[list[str], np.ndarray]:
    """120組番を辞書順で。polars の combo ソートと同じ並びになる。"""
    cs = sorted(
        f"{a}-{b}-{c}"
        for a in range(1, 7)
        for b in range(1, 7)
        for c in range(1, 7)
        if a != b and b != c and a != c
    )
    tri = np.array([[int(x[0]) - 1, int(x[2]) - 1, int(x[4]) - 1] for x in cs])
    return cs, tri


def boot_ci_paired(d: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    return d.mean(), np.quantile(d[idx].mean(axis=1), [0.025, 0.975])


def verdict(ll_a: np.ndarray, ll_b: np.ndarray, na: str, nb: str) -> float:
    """差・CI・判定を出し、**CI幅**を返す（検出力の値付けに使う）。"""
    diff, ci = boot_ci_paired(ll_a - ll_b)
    mark = "PASS" if ci[1] < 0 else "FAIL"
    w = ci[1] - ci[0]
    print(f"  {na} - {nb}: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  幅{w:.4f}  → {mark}")
    return w


def cv_clogit(X: np.ndarray, y: np.ndarray, w0: np.ndarray):
    """レース単位 5-fold CV。各 fold の外で推定した係数で評価する。"""
    n = len(y)
    fold = np.arange(n) % N_FOLDS
    ll = np.empty(n)
    ws = []
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        w = fit_clogit(X[tr], y[tr], w0)
        ll[te] = race_ll(X[te], y[te], w)
        ws.append(w)
    return ll, np.mean(ws, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stagewise",
        action="store_true",
        help="PL展開の代わりに stagewise_wf.parquet の段階条件付き確率を使う",
    )
    args = ap.parse_args()

    parsed = ROOT / "data" / "parsed"
    odds = pl.read_parquet(parsed / "odds3t.parquet")
    mkt = pl.read_parquet(parsed / "market_payout.parquet")  # 返還・同着 除外済み
    fund = pl.read_parquet(parsed.parent / "preds" / "fundamental_wf.parquet")

    # 板が完全なレース
    good = (
        odds.group_by(KEY)
        .agg(nn=pl.col("odds").is_null().sum(), n=pl.len(), lo=pl.col("odds").min())
        .filter((pl.col("nn") == 0) & (pl.col("n") == N_COMBO) & (pl.col("lo") > 0))
        .select(KEY)
    )
    # 6艇そろい・勝者1艇
    okf = (
        fund.group_by(KEY)
        .agg(n=pl.len(), w=pl.col("y_win").sum(), nn=pl.col("p_fund").is_null().sum())
        .filter((pl.col("n") == N_BOATS) & (pl.col("w") == 1) & (pl.col("nn") == 0))
        .select(KEY)
    )
    races = (
        good.join(okf, on=KEY)
        .join(mkt.select(KEY + ["combo"]), on=KEY)
        .pipe(walk_forward)  # 板は 2023年だけなので実質素通しだが、明示しておく
        .sort(KEY)
    )
    stage = None
    if args.stagewise:
        stage = pl.read_parquet(parsed.parent / "preds" / "stagewise_wf.parquet")
        races = races.join(stage.select(KEY), on=KEY).sort(KEY)
    print(f"封印境界 {HOLDOUT_START} — 板ありレースは全て walk-forward 側")
    print(f"対象レース: {races.height:,}  ({races['date'].min()}〜{races['date'].max()})")
    if races.height < 500:
        raise SystemExit("対象が少なすぎる。odds3t と払戻の重なりを確認せよ")

    n = races.height
    cs, tri = combo_list()
    pos = {c: i for i, c in enumerate(cs)}

    # 市場: 120通りの q
    o = odds.join(races.select(KEY), on=KEY).sort(KEY + ["combo"])
    assert o.height == n * N_COMBO, (o.height, n * N_COMBO)
    inv = (1 / o["odds"].to_numpy()).reshape(n, N_COMBO)
    q = inv / inv.sum(axis=1, keepdims=True)

    # fundamental: PL または段階条件付きモデルで 120通りへ展開
    f = fund.join(races.select(KEY), on=KEY).sort(KEY + ["boat"])
    s = f["p_fund"].to_numpy().reshape(n, N_BOATS)
    s = s / s.sum(axis=1, keepdims=True)
    if stage is None:
        sa, sb, sc = s[:, tri[:, 0]], s[:, tri[:, 1]], s[:, tri[:, 2]]
        p = sa * (sb / (1 - sa)) * (sc / (1 - sa - sb))
        fundamental_name = "PL"
    else:
        p = stage.join(races.select(KEY), on=KEY).sort(KEY)["p_stage"].to_numpy()
        fundamental_name = "段階条件付き"

    y = np.array([pos[c] for c in races["combo"].to_numpy()])

    lp = np.log(np.clip(p, 1e-12, None))
    lq = np.log(np.clip(q, 1e-12, None))
    X_mkt = lq[:, :, None]
    X_comb = np.stack([lp, lq], axis=2)

    ll_market = race_ll(X_mkt, y, np.array([1.0]))
    ll_fund = race_ll(lp[:, :, None], y, np.array([1.0]))
    ll_pow, w_pow = cv_clogit(X_mkt, y, np.array([1.0]))
    ll_comb, w_comb = cv_clogit(X_comb, y, np.array([0.0, 1.0]))

    print(f"\n係数（{N_FOLDS}-fold平均・条件付きロジット・120肢）")
    print(f"  market^b : b={w_pow[0]:.3f}")
    print(f"  combined : a(fundamental)={w_comb[0]:.3f}  b(market)={w_comb[1]:.3f}")

    print(f"\n===== 120通り空間 log-loss  races={n:,} =====")
    print(f"  market   (b=1固定)  : {ll_market.mean():.4f}")
    print(f"  market^b (power補正) : {ll_pow.mean():.4f}")
    print(f"  fundamental 単体({fundamental_name}) : {ll_fund.mean():.4f}")
    print(f"  combined             : {ll_comb.mean():.4f}")

    print("\n===== 判定（CI上限<0で合格）=====")
    w120 = verdict(ll_comb, ll_market, "combined", "market  ")
    print("      ↑ G2 本体（120通り空間版）")
    d_net, w_net = verdict(ll_comb, ll_pow, "combined", "market^b"), None
    print("      ↑ power補正を差し引いた fundamental の正味の寄与。**ここが本質**")
    verdict(ll_pow, ll_market, "market^b", "market  ")
    print("      ↑ 参考: 市場を b 乗するだけで得られる分（fundamental は無関係）")

    # ---- 同じレースで6艇空間の G2 も回して並べる ----
    mk6 = market_prob(odds.join(races.select(KEY), on=KEY))
    d6 = (
        f.join(mk6.select(KEY + ["boat", "q"]), on=KEY + ["boat"], how="inner")
        .sort(KEY + ["boat"])
    )
    n6 = d6.height // N_BOATS
    lf6 = np.log(np.clip(d6["p_fund"].to_numpy(), 1e-9, None)).reshape(n6, N_BOATS)
    lq6 = np.log(np.clip(d6["q"].to_numpy(), 1e-9, None)).reshape(n6, N_BOATS)
    y6 = d6["y_win"].to_numpy().reshape(n6, N_BOATS).argmax(axis=1)
    ll_m6 = race_ll(lq6[:, :, None], y6, np.array([1.0]))
    ll_c6, w6 = cv_clogit(np.stack([lf6, lq6], axis=2), y6, np.array([0.0, 1.0]))

    print(f"\n===== 参考: 同じレースで6艇空間の G2  races={n6:,} =====")
    print(f"  combined : a={w6[0]:.3f}  b={w6[1]:.3f}")
    print(f"  market {ll_m6.mean():.4f}  combined {ll_c6.mean():.4f}")
    w6ci = verdict(ll_c6, ll_m6, "combined", "market  ")

    print("\n===== 検出力（板を買い足す判断のため）=====")
    print(f"  6艇空間   の G2本体 CI幅: {w6ci:.4f}")
    print(f"  120通り空間の G2本体 CI幅: {w120:.4f}  （{w120 / w6ci:.1f}倍）")
    print(f"  CI幅は √n で縮む。板を {n:,} → N に増やしたときの G2本体 CI幅:")
    for mult in (2, 4, 10, 20):
        print(f"    {n * mult:>7,}レース: {w120 / np.sqrt(mult):.4f}")

    # 買い足す価値は「combined - market^b」を有意にできるかで決まる。
    # G2本体はすでに通っているが、その中身は power 補正であって fundamental ではない。
    d = (ll_comb - ll_pow).mean()
    h = d_net / 2  # 現在の片側幅
    need = n * (h / abs(d)) ** 2 if d < 0 else float("inf")
    print("\n  正味の寄与（combined - market^b）を有意にするのに必要な板の枚数:")
    print(f"    現在: 点推定 {d:+.4f} / 片側幅 {h:.4f} / {n:,}レース")
    if np.isfinite(need):
        print(f"    → 約 {need:,.0f}レース（今の点推定が真値なら）。{need / n:.1f}倍")
    else:
        print("    → 点推定が正（市場^b より悪い）。枚数を増やしても通らない")
    print(f"    ※ これは今の {fundamental_name} fundamental の点推定に基づく概算")


if __name__ == "__main__":
    main()
