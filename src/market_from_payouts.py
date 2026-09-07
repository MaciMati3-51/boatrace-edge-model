"""払戻から市場含意確率 q を復元する（スクレイピング不要・全期間）。

odds3t のオッズ板は規約上の理由で 5,310レース分しか無い（→ README §3）。
だが **log-loss は「実際に起きた目に割り当てた確率」だけで決まる** ので、
的中目の q さえ分かれば市場の log-loss は全期間で計算できる。

的中目の q は K文書の払戻から出せる:

    q_realized = (1 / (払戻金/100)) / Σ_c (1/odds_c)

分母は本来120通り全部が要るが、控除率が固定なので**ほぼ定数**。手持ちの完全板
5,238レースで実測すると Σ(1/odds) = 1.3364 ± 0.0018（sd）。定数で代用したときの
log 誤差は sd 0.0014・平均 0.0003 で、レースごとの加算定数として入るだけなので
20万レースで平均すれば SE は 1e-5 未満。無視できる。

## 単勝プールは使えない（2026-09-07 実測で棄却）

単勝の払戻からも同じ理屈で勝率レベルの q が出せる（6艇空間なので既存の G1/G2 と
直接比較できる）ため先に検討したが、完全板のあるレースで
`log(1/o_単勝) - log(q_3連単周辺化)` を測ると **sd 0.40** と大きくばらついた。
単勝プールが薄く、市場推定として質が違う。平均は +0.228（= 正規化子 0.796 ≒
法定の単勝払戻率 80%）なので理屈は合っているが、レース単位のノイズが検定したい
効果量（log-loss で 0.001〜0.003）を桁で上回る。よって**3連単プールのみ使う**。

## 除外するレース

- **返還**: F(フライング)・L(出遅れ)・K(欠場) を含むレース。当該艇の舟券が返還され、
  プールが縮んだ状態で払戻が決まる。「購入判定時のオッズ」と「清算の払戻」が
  別物になる（--keep-refund で残せる）。
- **同着**: ３連単の払戻行が2本以上あるレース。プールが的中目に分割されるので
  払戻から逆算した odds が実際の倍率とずれる。

この2つを除くと、板ありレースでの復元誤差は |log誤差|<0.01 が 100%（下の検証出力）。

出力: data/parsed/market_payout.parquet
実行: uv run python src/market_from_payouts.py
"""

import argparse

import numpy as np
import polars as pl

from train_fundamental import ROOT

KEY = ["date", "jcd", "race_no"]
N_COMBO = 120
REFUND_TOKENS = ["F", "L0", "L1", "K0", "K1"]


def board_normalizer(odds: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    """完全な120通り板から Σ(1/odds) を実測し、代表値（中央値）を返す。"""
    bad = (
        odds.group_by(KEY)
        .agg(nn=pl.col("odds").is_null().sum(), n=pl.len())
        .filter((pl.col("nn") > 0) | (pl.col("n") != N_COMBO))
        .select(KEY)
    )
    full = odds.join(bad, on=KEY, how="anti").filter(pl.col("odds") > 0)
    s = full.group_by(KEY).agg(S=(1 / pl.col("odds")).sum())
    v = s["S"].to_numpy()
    print(f"完全板レース: {len(v):,}")
    print(
        f"  Σ(1/odds)  mean {v.mean():.4f}  sd {v.std():.4f}  "
        f"min {v.min():.4f}  p50 {np.median(v):.4f}  max {v.max():.4f}"
    )
    return float(np.median(v)), full


def validate(full: pl.DataFrame, mkt: pl.DataFrame, s3: float, tag: str) -> None:
    """完全板のあるレースで、復元した q を板由来の真の q と突き合わせる。"""
    true_q = (
        full.with_columns(inv=1 / pl.col("odds"))
        .with_columns(q_true=pl.col("inv") / pl.col("inv").sum().over(KEY))
        .select(KEY + ["combo", "q_true"])
    )
    j = mkt.join(true_q, on=KEY + ["combo"], how="inner")
    if j.height == 0:
        print(f"\n[検証/{tag}] 板と払戻の重なりが無い。日付範囲を確認せよ")
        return
    e = np.log(j["q_3t"].to_numpy()) - np.log(j["q_true"].to_numpy())
    ae = np.abs(e)
    print(f"\n[検証/{tag}] 板ありレースで復元 q と真の q を比較: {j.height:,}件")
    print(
        f"  log誤差  mean {e.mean():+.5f}  sd {e.std():.5f}  "
        f"p1 {np.quantile(e, 0.01):+.4f}  p99 {np.quantile(e, 0.99):+.4f}"
    )
    print(f"  |log誤差| < 0.01 の割合: {(ae < 0.01).mean():.3%}")
    print(f"  Σ(1/odds) を定数 {s3:.4f} で代用したことによる誤差がこれ。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-refund",
        action="store_true",
        help="返還レース（F/L/K を含む）も残す。既定は除外",
    )
    args = ap.parse_args()

    parsed = ROOT / "data" / "parsed"
    odds = pl.read_parquet(parsed / "odds3t.parquet")
    s3, full = board_normalizer(odds)

    pay = pl.read_parquet(parsed / "payouts.parquet")
    mkt = (
        pay.filter((pl.col("bet_type") == "３連単") & (pl.col("amount") > 0))
        .with_columns(o_3t=pl.col("amount") / 100)
        .with_columns(q_3t=(1 / pl.col("o_3t")) / s3)
        .select(KEY + ["combo", "o_3t", "q_3t", "ninki"])
    )
    print(f"\n３連単の払戻行: {mkt.height:,}")
    validate(full, mkt, s3, "除外前")

    # 同着（３連単の払戻行が複数ある）レースを落とす
    dead_heat = mkt.group_by(KEY).len().filter(pl.col("len") > 1).select(KEY)
    print(f"\n同着レース: {dead_heat.height:,}")
    mkt = mkt.join(dead_heat, on=KEY, how="anti")

    res = pl.read_parquet(parsed / "results.parquet")
    refund = (
        res.filter(pl.col("rank_raw").is_in(REFUND_TOKENS)).select(KEY).unique()
    )
    print(f"\n返還レース（F/L/K を含む）: {refund.height:,}")
    mkt = mkt.join(refund.with_columns(refund=pl.lit(True)), on=KEY, how="left")
    mkt = mkt.with_columns(refund=pl.col("refund").fill_null(False))
    if not args.keep_refund:
        mkt = mkt.filter(~pl.col("refund"))
    validate(full, mkt, s3, "除外後")

    out = parsed / "market_payout.parquet"
    mkt.write_parquet(out)
    print(f"\n書き出し: {out}  {mkt.height:,}レース")
    print(f"  期間 {mkt['date'].min()}〜{mkt['date'].max()}")
    print(f"  正規化子 s3 = {s3:.4f}")


if __name__ == "__main__":
    main()
