"""fundamental モデル用の特徴量テーブルを構築する。

**時刻安全性が最優先**。各特徴量の確定時点:

  [前日確定]   B由来: 枠・級別・年齢・体重・全国/当地勝率・モーター/ボート2率
  [直前確定]   K由来: 展示タイム（周回展示は締切の約30分前に実施）
  [履歴]       選手ローリング統計 … **前日以前の結果のみ**（expanding + shift）
  [リーク禁止] K由来: 実進入コース・実ST・着順 → ラベル/検証専用。特徴量に入れない
               （実進入は発走時に確定する。予測に使えるのはスタート展示の進入のみ
                → beforeinfo 収集後に追加）

出力: data/features/base.parquet  1行 = (date, jcd, race_no, boat)
"""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
OUT = ROOT / "data" / "features"

CLASS_ORD = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def racer_history(results: pl.DataFrame) -> pl.DataFrame:
    """選手×日付ごとに「その日より前」の成績を expanding で集計。

    shift(1) は日単位で行う（同日内の別レースは互いに見えない扱い。
    同日リークを厳密に絶つため、レース単位でなく日単位でずらす）。
    """
    daily = (
        results.group_by(["racer_id", "date"])
        .agg(
            n=pl.len(),
            wins=(pl.col("rank_raw") == "01").sum(),
            top2=pl.col("rank_raw").is_in(["01", "02"]).sum(),
            st_sum=pl.col("st_num").sum(),
            st_n=pl.col("st_num").is_not_null().sum(),
            f=pl.col("rank_raw").str.starts_with("F").sum(),
        )
        .sort(["racer_id", "date"])
    )
    return daily.with_columns(
        [
            pl.col(c).cum_sum().shift(1).over("racer_id").alias(f"c_{c}")
            for c in ["n", "wins", "top2", "st_sum", "st_n", "f"]
        ]
    ).select(
        "racer_id",
        "date",
        hist_n=pl.col("c_n"),
        hist_win_rate=pl.col("c_wins") / pl.col("c_n"),
        hist_top2_rate=pl.col("c_top2") / pl.col("c_n"),
        hist_mean_st=pl.col("c_st_sum") / pl.col("c_st_n"),
        hist_f_rate=pl.col("c_f") / pl.col("c_n"),
    )


def venue_boat_history(results: pl.DataFrame) -> pl.DataFrame:
    """会場×枠の1着率を日付 expanding で（その日より前のみ）。"""
    daily = (
        results.group_by(["jcd", "boat", "date"])
        .agg(n=pl.len(), wins=(pl.col("rank_raw") == "01").sum())
        .sort(["jcd", "boat", "date"])
    )
    return daily.with_columns(
        cn=pl.col("n").cum_sum().shift(1).over(["jcd", "boat"]),
        cw=pl.col("wins").cum_sum().shift(1).over(["jcd", "boat"]),
    ).select(
        "jcd",
        "boat",
        "date",
        vb_win_rate=pl.col("cw") / pl.col("cn"),
        vb_n=pl.col("cn"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = pl.read_parquet(PARSED / "entries.parquet")
    results = pl.read_parquet(PARSED / "results.parquet")

    # ST を数値化（'0.21'→0.21、'F0.03'/'L'等は null。F は別特徴で拾う）
    results = results.with_columns(
        st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64)
    )

    base = entries.join(
        results.select(
            ["date", "jcd", "race_no", "boat", "tenji", "rank_raw", "entry_course", "st_num"]
        ),
        on=["date", "jcd", "race_no", "boat"],
        how="inner",  # 開催されたレースのみ
    )

    rh = racer_history(results)
    vb = venue_boat_history(results)

    feat = (
        base.join(rh, on=["racer_id", "date"], how="left")
        .join(vb, on=["jcd", "boat", "date"], how="left")
        .with_columns(
            class_ord=pl.col("class").replace_strict(CLASS_ORD, default=None),
            # 展示タイムはレース内相対値が本質（艇体差 > 絶対値）
            tenji_rank=pl.col("tenji")
            .rank("ordinal")
            .over(["date", "jcd", "race_no"]),
            tenji_rel=(
                pl.col("tenji")
                - pl.col("tenji").mean().over(["date", "jcd", "race_no"])
            ),
            # ラベル
            y_win=(pl.col("rank_raw") == "01").cast(pl.Int8),
            y_rank=pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8),
        )
    )
    feat.write_parquet(OUT / "base.parquet")
    print(f"rows={len(feat)} races={feat.select(['date','jcd','race_no']).unique().height}")
    print("null率:", {c: round(feat[c].null_count()/len(feat), 3)
                     for c in ["hist_win_rate", "vb_win_rate", "tenji", "class_ord"]})


if __name__ == "__main__":
    main()
