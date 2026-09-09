"""1着段（fundamental）が市場に負けている原因を分解する。

## 結果（2026-09-09）— 原因は「相手を見ていない」ことだった

このスクリプトが出した診断に従って `train_fundamental.py` をレース単位6クラス
softmax に置き換えた。以下の数値は**すべて旧実装（艇ごと二値→正規化）のときのもの**で、
再実行すれば新実装の値が出る。対応は:

    1着段の穴（板4,901R）  +0.0850 → +0.0613
    市場と本命が割れる率    18.9%  → 13.1%（割れたときの的中は 25.9% vs 市場 39.4% のまま）
    枠1の A1-B2 スプレッド  実物の36% → 93%

## 何を疑って書いたか

段階条件付きモデルで2・3着の穴を73%縮めた結果、120通り空間に残った穴 +0.1676 の
うち **約半分が1着段** になった（1着 +0.0849 / 条件付き +0.0827）。1着段は G1 を
通した 2026-07-19 以降まったく手を入れていない。ここが本当に負けているのか、
負けているなら原因は「情報が足りない」のか「持っている情報を使えていない」のかを
決める。

## 測り方の要点

- 市場の1着周辺確率 q は板（20230701〜0803 の4,901レース）を1着で周辺化して作る。
- **`KL(q||p)` を主な計測器にする。** 実現 log-loss 差は勝者1艇しか使わないので
  CI が ±0.0112 と広いが、市場をほぼ真とみなせば同じ量が `KL(q||p)` で書けて
  CI は ±0.0021 になる（実測: 実現差 +0.0850 / KL +0.0829 で一致）。条件別に
  切っても壊れない精度が出る。
- **枠×級別のバイアスは市場も板も要らない。** walk-forward 147,922レースの実勝率と
  平均予測を比べるだけ。板の単月バイアスと無関係に効く。

実行: uv run python src/diagnose_first_stage.py [--skip-fit]
"""

import argparse

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from holdout import walk_forward
from train_fundamental import CATEGORICAL, FEATURES, ROOT

KEY = ["date", "jcd", "race_no"]
N_BOATS = 6
SPLIT = "20240901"      # 修正案の検証用。学習=これ以前 / 検証=これ以降〜封印境界
BOARD_START = "20230701"


def ci(x: np.ndarray) -> float:
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """完走6艇レースの fundamental 予測・特徴量・板由来の市場1着確率。"""
    fund = pl.read_parquet(ROOT / "data/preds/fundamental_wf.parquet")
    ok = (
        fund.group_by(KEY)
        .agg(n=pl.len(), w=pl.col("y_win").sum(), nn=pl.col("p_fund").is_null().sum())
        .filter((pl.col("n") == N_BOATS) & (pl.col("w") == 1) & (pl.col("nn") == 0))
        .select(KEY)
    )
    fund = fund.join(ok, on=KEY)
    feat = pl.read_parquet(ROOT / "data/features/base.parquet").join(ok, on=KEY)

    odds = pl.read_parquet(ROOT / "data/parsed/odds3t.parquet")
    bad = (
        odds.group_by(KEY)
        .agg(nn=pl.col("odds").is_null().sum(), k=pl.len())
        .filter((pl.col("nn") > 0) | (pl.col("k") != 120))
        .select(KEY)
    )
    qwin = (
        odds.join(bad, on=KEY, how="anti")
        .filter(pl.col("odds") > 0)
        .with_columns(inv=1 / pl.col("odds"), boat=pl.col("combo").str.slice(0, 1).cast(pl.Int64))
        .group_by(KEY + ["boat"])
        .agg(m=pl.col("inv").sum())
        .with_columns(q=pl.col("m") / pl.col("m").sum().over(KEY))
        .select(KEY + ["boat", "q"])
    )
    return fund, feat, qwin


def races_of(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(KEY).unique(maintain_order=True)


def mat(df: pl.DataFrame, col: str, n: int) -> np.ndarray:
    return df[col].to_numpy().astype(float).reshape(n, N_BOATS)


def norm(m: np.ndarray) -> np.ndarray:
    return m / m.sum(axis=1, keepdims=True)


def section_board(fund: pl.DataFrame, qwin: pl.DataFrame):
    """板 4,901レースでの1着 log-loss と、KL を計測器として使ってよいかの検算。"""
    b = fund.join(qwin, on=KEY + ["boat"], how="inner").sort(KEY + ["boat"])
    n = b.height // N_BOATS
    P, Q, NV = norm(mat(b, "p_fund", n)), norm(mat(b, "q", n)), norm(mat(b, "p_naive", n))
    Y = mat(b, "y_win", n)
    w = Y.argmax(1)
    r = np.arange(n)
    ll = lambda M: -np.log(np.clip(M[r, w], 1e-12, None))
    llf, llq, lln = ll(P), ll(Q), ll(NV)
    kl = (Q * np.log(np.clip(Q, 1e-12, None) / np.clip(P, 1e-12, None))).sum(1)

    print(f"\n===== 1着(6艇空間) log-loss  板 {n:,}レース =====")
    print(f"  素朴(会場×枠) : {lln.mean():.4f}")
    print(f"  fundamental   : {llf.mean():.4f}")
    print(f"  市場(板を周辺化): {llq.mean():.4f}")
    print(f"  fundamental - 市場: {(llf - llq).mean():+.4f} ±{ci(llf - llq):.4f}  ← これが1着段の穴")
    print(f"  素朴→市場のうち fundamental が取れた割合: "
          f"{(lln.mean() - llf.mean()) / (lln.mean() - llq.mean()):.0%}")
    print(f"\n  [計測器の検算] 実現差 {(llf - llq).mean():+.4f} ±{ci(llf - llq):.4f} "
          f"vs KL(q||p) {kl.mean():+.4f} ±{ci(kl):.4f}")
    print("   一致する＝以降は結果ノイズ抜きで条件別に切ってよい（CIが5倍狭い）")

    pf, qf = P.argmax(1), Q.argmax(1)
    dis = pf != qf
    hm, hq = (w[dis] == pf[dis]).sum(), (w[dis] == qf[dis]).sum()
    print(f"\n===== 一番手が市場と割れたとき、どちらが正しいか =====")
    print(f"  割れた {dis.sum()}R ({dis.mean():.1%}): "
          f"モデルの本命 {hm / dis.sum():.1%} 的中 / 市場の本命 {hq / dis.sum():.1%} 的中")
    print(f"  McNemar z = {(hq - hm) / np.sqrt(hq + hm):.2f}（市場優位）")
    print("   → モデルが市場と違う見立てをするとき、それは新情報ではなく単なる誤り")
    return b, P, Q, Y, kl


def section_bias(fund: pl.DataFrame, feat: pl.DataFrame):
    """市場を使わずに 147,922レースで測る。枠×級別の予測バイアスと resolution。"""
    d = (
        walk_forward(fund)
        .join(feat.select(KEY + ["boat", "class"]), on=KEY + ["boat"], how="inner")
        .sort(KEY + ["boat"])
    )
    n = d.height // N_BOATS
    P, Y = norm(mat(d, "p_fund", n)), mat(d, "y_win", n)
    print(f"\n===== 枠×級別のバイアス（walk-forward {n:,}レース・市場も板も不要）=====")
    t = (
        d.group_by(["boat", "class"])
        .agg(k=pl.len(), actual=pl.col("y_win").mean(), pred=pl.col("p_fund").mean())
        .filter(pl.col("k") >= 300)
        .sort(["boat", "class"])
    )
    print("  枠 級別       n      実勝率    モデル      差")
    for row in t.iter_rows(named=True):
        print(f"  {row['boat']}  {row['class']:3}  {row['k']:7,}  {row['actual']:7.2%}  "
              f"{row['pred']:7.2%}  {row['pred'] - row['actual']:+7.2%}")
    a = t.filter((pl.col("boat") == 1) & (pl.col("class") == "A1")).row(0, named=True)
    z = t.filter((pl.col("boat") == 1) & (pl.col("class") == "B2")).row(0, named=True)
    print(f"\n  枠1の A1−B2 スプレッド: 実測 {a['actual'] - z['actual']:+.1%} / "
          f"モデル {a['pred'] - z['pred']:+.1%} → モデルは実物の "
          f"{(a['pred'] - z['pred']) / (a['actual'] - z['actual']):.0%} しか出していない")

    print(f"\n  枠1: 平均予測 {P[:, 0].mean():.4f} / 実勝率 {Y[:, 0].mean():.4f}"
          "  ← 平均は合っている（calibration ではなく resolution の欠損）")
    q = np.quantile(P[:, 0], np.linspace(0, 1, 11))
    q[0] -= 1e-9
    bins = np.digitize(P[:, 0], q[1:-1])
    print("  枠1予測の十分位:  モデル → 実勝率")
    for i in range(10):
        m = bins == i
        print(f"   {i + 1:2d}: {P[m, 0].mean():6.2%} → {Y[m, 0].mean():6.2%}  (n={m.sum():,})")
    print("   両端が内側に潰れている＝「枠1は55%前後」という枠の事前分布に張り付いている")


def section_slices(b: pl.DataFrame, kl: np.ndarray):
    """条件別の KL。どの状況で market に離されているか。"""
    keys = races_of(b)
    n = keys.height
    res = pl.read_parquet(ROOT / "data/parsed/results.parquet").join(keys, on=KEY, how="inner")
    cond = res.group_by(KEY).agg(
        wind=pl.col("wind_speed").first(),
        kimarite=pl.col("kimarite").drop_nulls().first(),
        n_change=(pl.col("entry_course") != pl.col("boat")).sum(),
    )
    feat = pl.read_parquet(ROOT / "data/features/base.parquet").join(keys, on=KEY, how="inner")
    cls1 = feat.group_by(KEY).agg(cls1=pl.col("class").filter(pl.col("boat") == 1).first())
    df = (
        keys.with_columns(pl.Series("kl", kl))
        .join(cond, on=KEY, how="left")
        .join(cls1, on=KEY, how="left")
    )
    print(f"\n===== 条件別の穴 KL(q||p)（全体 {kl.mean():+.4f}）=====")
    for col, label in [("n_change", "進入変更した艇数（実進入≠枠）"), ("cls1", "1号艇の級別"),
                       ("kimarite", "決まり手（事後）"), ("wind", "風速(m)")]:
        s = (df.group_by(col).agg(n=pl.len(), kl=pl.col("kl").mean())
             .filter(pl.col("n") >= 30).sort("kl", descending=True))
        print(f"  --- {label} ---")
        for row in s.iter_rows(named=True):
            print(f"    {str(row[col]):>8}  n={row['n']:5,}  KL={row['kl']:+.4f}")
    o = np.sort(kl)[::-1]
    print("  --- 集中度 ---")
    for f in (0.10, 0.25, 0.50):
        k = int(n * f)
        print(f"    上位{f:4.0%}のレースが全KLの {o[:k].sum() / kl.sum():5.1%}")
    print("    穴は一部の壊れたレースではなく全体に薄く広がっている")


def section_residual(b: pl.DataFrame, P: np.ndarray, Q: np.ndarray):
    """log(q/p) は手持ちの特徴量で説明できるか。できるなら不足は情報でなく学習。"""
    n = P.shape[0]
    feat = pl.read_parquet(ROOT / "data/features/base.parquet")
    j = b.select(KEY + ["boat"]).join(feat, on=KEY + ["boat"], how="left").sort(KEY + ["boat"])
    X = j.select(FEATURES).to_numpy().astype(float)
    R = np.log(Q) - np.log(P)
    R -= R.mean(1, keepdims=True)
    y = R.reshape(-1)
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    groups = np.repeat(np.arange(n), N_BOATS)
    oof = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=50, categorical_features=cat,
                                          random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    rhat = oof.reshape(n, N_BOATS)
    rhat -= rhat.mean(1, keepdims=True)
    kl0 = (Q * np.log(Q / P)).sum(1)
    print("\n===== 残差 log(q_市場/p_モデル) は手持ちの特徴量で予測できるか =====")
    print(f"  out-of-fold R^2 = {1 - ((y - oof) ** 2).sum() / (y ** 2).sum():.3f}"
          f"   相関 = {np.corrcoef(y, oof)[0, 1]:.3f}")
    for lam in (0.5, 1.0):
        z = np.log(P) + lam * rhat
        z -= z.max(1, keepdims=True)
        Pn = norm(np.exp(z))
        kl = (Q * np.log(Q / Pn)).sum(1)
        print(f"  λ={lam}: KL {kl0.mean():.4f} → {kl.mean():.4f}（{1 - kl.mean() / kl0.mean():.0%} 縮小）")
    print("  → 市場との差の半分は、我々が既に持っているデータの中に書いてある")


def section_fix(fund: pl.DataFrame, feat: pl.DataFrame, qwin: pl.DataFrame):
    """原因の検証。「艇ごと二値→正規化」を「レース単位6クラス softmax」に替える。

    6艇ぶんの特徴量を1行にまとめるので、モデルが相手を見られる。出力は最初から
    レース内で和が1（正規化が目的関数の内側に入る）。
    """
    d = walk_forward(feat).sort(KEY + ["boat"])
    races = races_of(d)
    n = races.height
    X = d.select(FEATURES).to_numpy().astype(float).reshape(n, N_BOATS, -1)
    Y = mat(d, "y_win", n)
    y = Y.argmax(1)
    dates = races["date"].to_numpy()
    keep = [i for i, c in enumerate(FEATURES) if c != "boat"]  # 枠は列の位置で表現される
    Xr = np.hstack([X[:, 0, FEATURES.index("jcd")][:, None], X[:, :, keep].reshape(n, -1)])

    f = walk_forward(fund).sort(KEY + ["boat"])
    P = norm(mat(f, "p_fund", n))

    for train_end, label, board in [(SPLIT, f"時間分割 学習〜{SPLIT}", False),
                                    (BOARD_START, f"板と同条件 学習〜{BOARD_START}", True)]:
        tr = dates < train_end
        if board:
            bk = races_of(qwin.join(races, on=KEY)).with_columns(f=pl.lit(True))
            te = races.join(bk, on=KEY, how="left")["f"].fill_null(False).to_numpy()
        else:
            te = ~tr
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                             min_samples_leaf=50, categorical_features=[0],
                                             random_state=0).fit(Xr[tr], y[tr])
        Pm = clf.predict_proba(Xr[te])

        Pc = P.copy()
        for i in range(N_BOATS):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            Pc[:, i] = iso.fit(P[tr, i], Y[tr, i]).predict(P[:, i])
        Pc = norm(Pc)

        r = np.arange(te.sum())
        yt = y[te]
        ll = lambda M: -np.log(np.clip(M[r, yt], 1e-12, None))
        base, iso_ll, new = ll(P[te]), ll(Pc[te]), ll(Pm)
        print(f"\n===== 修正案の検証 [{label}] 検証 {te.sum():,}レース =====")
        print(f"  現行 fundamental        : {base.mean():.4f}")
        print(f"  + 枠別 isotonic 再較正   : {iso_ll.mean():.4f}  "
              f"差 {(iso_ll - base).mean():+.4f} ±{ci(iso_ll - base):.4f}")
        print(f"  レース単位6クラス softmax: {new.mean():.4f}  "
              f"差 {(new - base).mean():+.4f} ±{ci(new - base):.4f}")
        if board:
            bb = fund.join(qwin, on=KEY + ["boat"], how="inner").sort(KEY + ["boat"])
            m = bb.height // N_BOATS
            Q = norm(mat(bb, "q", m))
            assert races_of(bb).equals(races.filter(pl.Series(te))) and m == len(Pm)
            llq = -np.log(np.clip(Q[np.arange(m), yt], 1e-12, None))
            print(f"  市場                    : {llq.mean():.4f}")
            print(f"  → 穴 {(base - llq).mean():+.4f} → {(new - llq).mean():+.4f}"
                  f"（{1 - (new - llq).mean() / (base - llq).mean():.0%} 縮小）")
            for name, M in [("現行", P[te]), ("6クラス", Pm)]:
                dis = M.argmax(1) != Q.argmax(1)
                print(f"    {name:7}: 市場と本命が割れる {dis.mean():5.1%}  "
                      f"そこでの的中 モデル {(yt[dis] == M[dis].argmax(1)).mean():5.1%} / "
                      f"市場 {(yt[dis] == Q[dis].argmax(1)).mean():5.1%}")
        for cc in ("A1", "B1", "B2"):
            cl = d.filter(pl.col("boat") == 1)["class"].to_numpy()[te] == cc
            if cl.sum() < 200:
                continue
            print(f"    枠1 {cc}: 実 {Y[te][cl, 0].mean():6.2%} / 現行 {P[te][cl, 0].mean():6.2%}"
                  f" / 6クラス {Pm[cl, 0].mean():6.2%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fit", action="store_true", help="学習を伴う節（残差・修正案）を飛ばす")
    args = ap.parse_args()

    fund, feat, qwin = load()
    board_fund = fund.join(races_of(qwin), on=KEY, how="inner")
    print(f"板 {races_of(board_fund).height:,}レース "
          f"({board_fund['date'].min()}〜{board_fund['date'].max()}) はすべて walk-forward 側")

    b, P, Q, Y, kl = section_board(board_fund, qwin)
    section_bias(fund, feat)
    section_slices(b, kl)
    if not args.skip_fit:
        section_residual(b, P, Q)
        section_fix(fund, feat, qwin)


if __name__ == "__main__":
    main()
