"""2019-07 より前の公式 B/K を、本体とは別のファイルに取り込む（v4 以降の学習・履歴用）。

**本体の data/parsed/{entries,results,payouts}.parquet には混ぜない。** v3（凍結・前向き検証中）の特徴量は
「20190701 以降のデータで作った履歴」が前提で、v3 の予測経路は本体の parquet を丸ごと読む。古い年を本体に
足すと v3 の通算成績などが学習時と違う分布になる（v2 で 2022→2019 に遡ったときと同じ問題）。

入力: data/raw/txt_ext/{B,K}/*/*.TXT（download_lzh.py → extract_lzh_ext.py で用意する。
      本体の data/raw/txt/ には置かない。extract_lzh.py・import_bk_range.py・parse_kb.py は使わない）
出力: data/parsed/ext/{entries,results,payouts,race_meta}.parquet

実行: .venv/bin/python src/build_history_ext.py
"""
from pathlib import Path

import polars as pl

from parse_b_meta import parse_file as parse_meta_file
from parse_kb import parse_b_file, parse_k_file

ROOT = Path(__file__).resolve().parents[1]
MAIN, EXT = ROOT / "data/parsed", ROOT / "data/parsed/ext"
CUT = "20190701"  # 本体の先頭日。これより前だけを取り込む


def _files(kind):
    for path in sorted((ROOT / f"data/raw/txt_ext/{kind}").glob("*/*.TXT")):
        date = "20" + path.stem[1:]
        if date < CUT:
            yield path, date


def _meta(rows):
    """parse_b_meta.main と同じ導出（締切の分・時間帯）。"""
    df = pl.DataFrame(rows).unique(subset=["date", "jcd", "race_no"], keep="first")
    mins = pl.col("deadline").str.slice(0, 2).cast(pl.Int64) * 60 + pl.col("deadline").str.slice(3, 2).cast(pl.Int64)
    df = df.with_columns(min_of_day=pl.when(mins < 300).then(mins + 1440).otherwise(mins))
    last = df.group_by(["date", "jcd"]).agg(last_min=pl.col("min_of_day").max(), first_min=pl.col("min_of_day").min())
    return df.join(last, on=["date", "jcd"]).with_columns(
        slot=pl.when(pl.col("last_min") >= 22 * 60).then(pl.lit("ミッドナイト"))
        .when(pl.col("last_min") >= 20 * 60).then(pl.lit("ナイター"))
        .when(pl.col("last_min") >= 17 * 60 + 30).then(pl.lit("サマータイム"))
        .when(pl.col("first_min") <= 10 * 60).then(pl.lit("モーニング"))
        .otherwise(pl.lit("デイ")))


def main():
    tables = {"entries": [], "results": [], "payouts": []}
    meta = []
    for path, date in _files("B"):
        tables["entries"].extend(parse_b_file(path, date))
        meta.extend(parse_meta_file(path))
    for path, date in _files("K"):
        r, p = parse_k_file(path, date)
        tables["results"].extend(r)
        tables["payouts"].extend(p)
    EXT.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        schema = pl.read_parquet_schema(MAIN / f"{name}.parquet")
        df = pl.DataFrame(rows, infer_schema_length=None).select(
            [pl.col(c).cast(t) for c, t in schema.items()]).sort("date", maintain_order=True)
        if df["date"].max() >= CUT:
            raise ValueError("Extension overlaps the main tables")
        df.write_parquet(EXT / f"{name}.parquet")
        print(f"{name}: {df.height:,} rows, {df['date'].min()}〜{df['date'].max()}", flush=True)
    schema = pl.read_parquet_schema(MAIN / "race_meta.parquet")
    df = _meta(meta).select([pl.col(c).cast(t) for c, t in schema.items()])
    df.write_parquet(EXT / "race_meta.parquet")
    print(f"race_meta: {df.height:,} races, {df['date'].min()}〜{df['date'].max()}")


if __name__ == "__main__":
    main()
