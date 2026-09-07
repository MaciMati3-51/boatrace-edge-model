"""策1: 市場と fundamental の差を、全期間 20万レースで測る（オッズ板を使わない）。

G2 予備判定は板のある 4,901レースでしかできず、CI は ±0.0016 だった。だが
「市場が fundamental をどれだけ引き離しているか」（＝埋めるべき穴の大きさ）は
的中目の q だけで測れる。log-loss は実際に起きた目の確率しか使わないからである。
q は market_from_payouts.py が払戻から復元済み（板ありレースで検証、
|log誤差|<0.01 が100%）。

## 比較する空間

**3連単の120通り空間**で測る。理由は2つ:

1. 復元した q は的中目（3連単の組番）の確率なので、この空間が素の単位である。
2. 単勝プールから勝率レベルの q を作る道も検討したが、板ありレースで
   `log(1/o_単勝) - log(q_3連単周辺化)` の sd が 0.40 と大きく、検定したい効果量
   （0.001〜0.003）を桁で上回る。棄却した（market_from_payouts.py 冒頭に記録）。

fundamental 側は Plackett-Luce で 6艇の強さ → 120通りへ展開する:

    P(a→b→c) = s_a · s_b/(1-s_a) · s_c/(1-s_a-s_b)

PL は IIA を仮定するので競艇では不利に働く（決まり手で2着以下の分布が変わる）。
つまりここで出る差は fundamental 側に**甘くない**。穴の大きさの上限側の評価になる。

## ここで測れないもの

G2 本体（合成 vs 市場）は測れない。合成 P ∝ p^a·q^b の正規化子 Σ_c p_c^a q_c^b に
全120通りの q が要るため。板が要る。**これは板を買う/集める判断の前段**であって
代わりではない。

実行: uv run python src/gap_at_scale.py
"""

import argparse

import numpy as np
import polars as pl

from train_fundamental import ROOT

KEY = ["date", "jcd", "race_no"]
N_BOATS = 6
N_COMBO = 120
CHUNK = 20_000


def combo_index() -> np.ndarray:
    """120通りの (1着,2着,3着) を 0-index の艇番で。組番文字列と同じ並びで作る。"""
    return np.array(
        [(a, b, c) for a in range(6) for b in range(6) for c in range(6)
         if a != b and b != c and a != c],
        dtype=np.int64,
    )


def pl_prob(s: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Plackett-Luce。s:(R,6) 行和1、tri:(K,3) → (R,K)。"""
    sa = s[:, tri[:, 0]]
    sb = s[:, tri[:, 1]]
    sc = s[:, tri[:, 2]]
    return sa * (sb / (1 - sa)) * (sc / (1 - sa - sb))


def paired_ci(d: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """対応のある差の平均と95%CI。n が大きいので解析CIを主、bootstrap を検算に使う。"""
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = d[rng.integers(0, len(d), len(d))].mean()
    return m, (m - 1.96 * se, m + 1.96 * se), tuple(np.quantile(bs, [0.025, 0.975]))


def report(a: np.ndarray, b: np.ndarray, na: str, nb: str) -> None:
    m, ci, bci = paired_ci(a - b)
    print(f"  {na} - {nb}: {m:+.4f}")
    print(f"      解析CI     [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"      bootstrapCI[{bci[0]:+.4f}, {bci[1]:+.4f}]")


def decompose(df: pl.DataFrame, s_f: np.ndarray, hit_tri: np.ndarray) -> None:
    """板のあるレースで、120通り空間の差の内訳を見る。

    120通りの log-loss 差は2つの原因が混ざっている:
      (a) 1着を当てる力の差            … fundamental 本体の弱さ
      (b) 「1着を所与とした2・3着」の差 … PL の IIA 仮定の破れ（決まり手を知らない）

    (a) は板があれば周辺化して分離できる。板は 5,000レースしか無いが、内訳の
    比率を知るには十分。ここで (b) が支配的なら、**次に効くのは fundamental の
    強化ではなく段階条件付きモデルへの置き換え**という話になる。
    """
    odds = pl.read_parquet(ROOT / "data" / "parsed" / "odds3t.parquet")
    bad = (
        odds.group_by(KEY)
        .agg(nn=pl.col("odds").is_null().sum(), k=pl.len())
        .filter((pl.col("nn") > 0) | (pl.col("k") != N_COMBO))
        .select(KEY)
    )
    odds = odds.join(bad, on=KEY, how="anti").filter(pl.col("odds") > 0)
    qwin = (
        odds.with_columns(
            inv=1 / pl.col("odds"),
            boat=pl.col("combo").str.slice(0, 1).cast(pl.Int64),
        )
        .group_by(KEY + ["boat"])
        .agg(m=pl.col("inv").sum())
        .with_columns(q_win=pl.col("m") / pl.col("m").sum().over(KEY))
        .select(KEY + ["boat", "q_win"])
    )
    # 勝者艇の q_win だけ要る
    idx = df.with_row_index("i").select(KEY + ["i", "q_3t"])
    win_boat = pl.DataFrame(
        {"i": np.arange(df.height, dtype=np.uint32), "boat": hit_tri[:, 0] + 1}
    )
    j = idx.join(win_boat, on="i").join(qwin, on=KEY + ["boat"], how="inner")
    if j.height == 0:
        print("\n[分解] 板のあるレースが突合しない。スキップ")
        return
    i = j["i"].to_numpy()
    ll_m6 = -np.log(j["q_win"].to_numpy())
    ll_m120 = -np.log(j["q_3t"].to_numpy())
    sa = s_f[i, hit_tri[i, 0]]
    sb = s_f[i, hit_tri[i, 1]]
    sc = s_f[i, hit_tri[i, 2]]
    ll_f6 = -np.log(sa)
    ll_f120 = -np.log(sa * (sb / (1 - sa)) * (sc / (1 - sa - sb)))
    g6 = (ll_f6 - ll_m6).mean()
    g120 = (ll_f120 - ll_m120).mean()
    print(f"\n===== 差の分解（板のある {len(i):,}レース）=====")
    print(f"  1着だけ(6艇空間) の差   : {g6:+.4f}")
    print(f"  120通り空間の差         : {g120:+.4f}")
    print(f"  → 2・3着の条件付き部分  : {g120 - g6:+.4f}  ({(g120 - g6) / g120:.0%})")
    print("  後者が大きいなら、効くのは fundamental 強化より段階条件付きモデル。")


def roi_topk(pfull: np.ndarray, hit: np.ndarray, payout: np.ndarray, ks) -> None:
    """オッズを使わない賭けルール（モデル上位k点買い）の回収率。

    払戻だけで**厳密に**計算できる。当たった目の払戻は K文書にあり、外れは掛け金を
    失うだけだから。オッズ板は一切要らない。
    """
    order = np.argsort(-pfull, axis=1)
    rank_of_hit = np.argmax(order == hit[:, None], axis=1)  # 的中目がモデル何位か
    print(f"\n===== オッズ非使用ルールの回収率（{len(hit):,}レース・100円/点）=====")
    print("  k   的中率    回収率")
    for k in ks:
        win = rank_of_hit < k
        ret = np.where(win, payout, 0.0).sum()
        cost = 100.0 * k * len(hit)
        print(f"  {k:<3} {win.mean():>7.3%}  {ret / cost:>7.2%}")
    print("  ※ 控除率25%＝上限75%。上回らないのは favorite-longshot bias の分。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-year", action="store_true", help="年別の内訳も出す")
    ap.add_argument("--decompose", action="store_true", help="板のあるレースで差の内訳を出す")
    args = ap.parse_args()

    fund = pl.read_parquet(ROOT / "data" / "preds" / "fundamental_wf.parquet")
    mkt = pl.read_parquet(ROOT / "data" / "parsed" / "market_payout.parquet")

    # 6艇そろい・勝者1艇のレースに限る
    ok = (
        fund.group_by(KEY)
        .agg(n=pl.len(), w=pl.col("y_win").sum(), nn=pl.col("p_fund").is_null().sum())
        .filter((pl.col("n") == N_BOATS) & (pl.col("w") == 1) & (pl.col("nn") == 0))
        .select(KEY)
    )
    fund = fund.join(ok, on=KEY).sort(KEY + ["boat"])

    df = (
        mkt.join(fund.select(KEY).unique(), on=KEY, how="inner")
        .sort(KEY)
    )
    print(f"fundamental 予測あり: {fund.select(KEY).unique().height:,}レース")
    print(f"払戻由来 q あり     : {mkt.height:,}レース")
    print(f"突合                : {df.height:,}レース")

    fund = fund.join(df.select(KEY), on=KEY).sort(KEY + ["boat"])
    n = df.height
    s_f = fund["p_fund"].to_numpy().reshape(n, N_BOATS)
    s_n = fund["p_naive"].to_numpy().reshape(n, N_BOATS)
    s_f = s_f / s_f.sum(axis=1, keepdims=True)
    s_n = s_n / s_n.sum(axis=1, keepdims=True)

    tri = combo_index()
    combo = df["combo"].to_numpy()
    hit_tri = np.array([[int(x[0]) - 1, int(x[2]) - 1, int(x[4]) - 1] for x in combo])
    hit_idx = np.array(
        [np.flatnonzero((tri == t).all(axis=1))[0] for t in hit_tri], dtype=np.int64
    )

    q = df["q_3t"].to_numpy()

    def realized_pl(s: np.ndarray) -> np.ndarray:
        a, b, c = hit_tri[:, 0], hit_tri[:, 1], hit_tri[:, 2]
        r = np.arange(n)
        sa, sb, sc = s[r, a], s[r, b], s[r, c]
        return sa * (sb / (1 - sa)) * (sc / (1 - sa - sb))

    p_f = realized_pl(s_f)
    p_n = realized_pl(s_n)

    ll_m = -np.log(np.clip(q, 1e-12, None))
    ll_f = -np.log(np.clip(p_f, 1e-12, None))
    ll_n = -np.log(np.clip(p_n, 1e-12, None))

    print(f"\n===== 3連単120通り空間の log-loss  races={n:,} =====")
    print(f"  市場（払戻由来 q）   : {ll_m.mean():.4f}")
    print(f"  fundamental (PL展開) : {ll_f.mean():.4f}")
    print(f"  素朴 (会場×枠, PL)   : {ll_n.mean():.4f}")
    print(f"  一様 (1/120)         : {np.log(120):.4f}")

    print("\n===== 差（負なら前者が良い）=====")
    report(ll_f, ll_m, "fundamental", "市場       ")
    print("      ↑ これが埋めるべき穴。G2 はこの差を市場側に足して覆せるかを問う")
    report(ll_f, ll_n, "fundamental", "素朴       ")
    print("      ↑ G1 相当（120通り空間版）")

    if args.by_year:
        year = np.array([d[:4] for d in df["date"].to_numpy()])
        print("\n===== 年別の穴（fundamental - 市場）=====")
        for y in sorted(set(year)):
            m = year == y
            d = (ll_f - ll_m)[m]
            se = d.std(ddof=1) / np.sqrt(m.sum())
            print(f"  {y}: {d.mean():+.4f} ±{1.96 * se:.4f}  ({m.sum():,}レース)")

    if args.decompose:
        decompose(df, s_f, hit_tri)

    # ---- 回収率（オッズ非使用ルール） ----
    payout = df["o_3t"].to_numpy() * 100.0
    pfull = np.empty((n, N_COMBO), dtype=np.float32)
    for i in range(0, n, CHUNK):
        pfull[i:i + CHUNK] = pl_prob(s_f[i:i + CHUNK], tri).astype(np.float32)
    roi_topk(pfull, hit_idx, payout, [1, 2, 3, 5, 10, 20, 40, 120])


if __name__ == "__main__":
    main()
