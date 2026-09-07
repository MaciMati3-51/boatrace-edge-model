"""第2段: fundamental × 市場含意確率の合成 → G2 判定（最重要ゲート）。

まず**勝率レベル**で判定する（120通り展開より先に、優位の有無だけを最短で見る）:

  市場含意勝率 q_i = Σ_{combo: 1着=i} (1/odds) を正規化したもの
  合成: 条件付きロジット z_i = a·log p_fund_i + b·log q_i → レース内 softmax
  G2:  合成の 1着者 log-loss が市場単体を有意に上回るか（レース単位ブートストラップ）

合成の学習も月単位 walk-forward（fundamental の OOS 予測のみ使用、二重に OOS）。

## 合成器に sklearn の LogisticRegression を使ってはいけない（2026-07-20 修正）

当初 `LogisticRegression` で推定していたが**これは誤り**。評価はレース内 softmax
（6艇から1艇が勝つ条件付き多項分布）なのに、LogisticRegression は艇ごとの**二値**
分類を最適化する別の目的関数で、両者は一致しない。結果として softmax 評価上は歪んだ
係数が選ばれ、**市場単体より悪い合成が出る**という本来ありえない現象が起きていた
（実測: b=1.381 が選ばれ log-loss が市場 1.1686 → 合成 1.2167 と悪化）。

条件付きロジット（McFadden。Benter が使ったのもこれ）なら a=0,b=1 で市場を厳密に
再現できるため、合成が市場を下回ることは原理的に起きない。実装は g2_prelim.py に
置いてある（先に予備判定で使ったため）。

実行: uv run --with scipy python src/combine_market.py
"""

from pathlib import Path

import numpy as np
import polars as pl

from g2_prelim import boot_ci, fit_clogit, race_ll

ROOT = Path(__file__).resolve().parent.parent
N_BOATS = 6


def main() -> None:
    fund = pl.read_parquet(ROOT / "data" / "preds" / "fundamental_wf.parquet")
    odds = pl.read_parquet(ROOT / "data" / "parsed" / "odds3t.parquet")

    # 市場含意勝率（3連単 120通り → 1着艇へ周辺化）
    odds = odds.filter(pl.col("odds").is_not_null() & (pl.col("odds") > 0))
    odds = odds.with_columns(
        inv=1 / pl.col("odds"),
        first=pl.col("combo").str.slice(0, 1).cast(pl.Int64),
    )
    mkt = (
        odds.group_by(["date", "jcd", "race_no", "first"])
        .agg(m=pl.col("inv").sum())
        .with_columns(
            q=pl.col("m") / pl.col("m").sum().over(["date", "jcd", "race_no"])
        )
        .rename({"first": "boat"})
    )

    df = fund.join(mkt.select(["date", "jcd", "race_no", "boat", "q"]),
                   on=["date", "jcd", "race_no", "boat"], how="inner")
    # 6艇そろっているレースのみ
    full = df.group_by(["date", "jcd", "race_no"]).len().filter(pl.col("len") == 6)
    df = df.join(full.select(["date", "jcd", "race_no"]), on=["date", "jcd", "race_no"])
    # 勝者がちょうど1艇のレースのみ（返還等を除外）
    one = (
        df.group_by(["date", "jcd", "race_no"])
        .agg(w=pl.col("y_win").sum())
        .filter(pl.col("w") == 1)
        .select(["date", "jcd", "race_no"])
    )
    df = (
        df.join(one, on=["date", "jcd", "race_no"])
        .with_columns(
            lf=pl.col("p_fund").clip(1e-9, 1).log(),
            lq=pl.col("q").clip(1e-9, 1).log(),
        )
        .sort(["date", "jcd", "race_no", "boat"])
    )
    print(f"共通レース数: {df.height // N_BOATS}")

    months = sorted(df["date"].str.slice(0, 6).unique().to_list())
    ll_c, ll_m, ll_f = [], [], []
    for m in months:
        tr = df.filter(pl.col("date").str.slice(0, 6) < m)
        te = df.filter(pl.col("date").str.slice(0, 6) == m)
        if tr.height // N_BOATS < 500 or te.height == 0:
            continue

        def arrs(d: pl.DataFrame):
            r = d.height // N_BOATS
            lf = d["lf"].to_numpy().reshape(r, N_BOATS)
            lq = d["lq"].to_numpy().reshape(r, N_BOATS)
            y = d["y_win"].to_numpy().reshape(r, N_BOATS).argmax(axis=1)
            return np.stack([lf, lq], axis=2), lf, lq, y

        X_tr, *_, y_tr = arrs(tr)
        X_te, lf_te, lq_te, y_te = arrs(te)
        w = fit_clogit(X_tr, y_tr, np.array([0.0, 1.0]))
        ll_c.append(race_ll(X_te, y_te, w))
        ll_m.append(race_ll(lq_te[:, :, None], y_te, np.array([1.0])))
        ll_f.append(race_ll(lf_te[:, :, None], y_te, np.array([1.0])))
        print(f"{m}: a(fund)={w[0]:.3f} b(mkt)={w[1]:.3f}", flush=True)

    if not ll_c:
        print("G2判定に足るデータがない")
        return
    ll_comb, ll_mkt, ll_fund = (np.concatenate(x) for x in (ll_c, ll_m, ll_f))
    print(f"\nraces={len(ll_comb)}")
    print(f"market log-loss:      {ll_mkt.mean():.4f}")
    print(f"fundamental log-loss: {ll_fund.mean():.4f}")
    print(f"combined log-loss:    {ll_comb.mean():.4f}")
    diff, ci = boot_ci(ll_comb - ll_mkt)
    print(f"combined - market: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print("G2", "PASS" if ci[1] < 0 else "FAIL", "(CI上限<0で合格)")


if __name__ == "__main__":
    main()
