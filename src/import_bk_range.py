"""ローカルの公式 TXT から指定期間だけを取り込む（他の日付の行は変えない）。

実行: .venv/bin/python src/import_bk_range.py 20190701 20210630
"""
import sys
from pathlib import Path

import polars as pl

from parse_kb import parse_b_file, parse_k_file

ROOT = Path(__file__).resolve().parents[1]


def main(lo, hi):
    for kind in ("B", "K"):
        collected = {"entries": []} if kind == "B" else {"results": [], "payouts": []}
        dates = []
        for path in sorted((ROOT / f"data/raw/txt/{kind}").glob("*/*.TXT")):
            date = "20" + path.stem[1:]
            if not lo <= date <= hi:
                continue
            if kind == "B":
                collected["entries"].extend(parse_b_file(path, date))
            else:
                r, p = parse_k_file(path, date)
                collected["results"].extend(r)
                collected["payouts"].extend(p)
            dates.append(date)
        for name, rows in collected.items():
            if not rows:  # 当日の K（結果）はまだ無い
                print(name, "no rows in range", flush=True)
                continue
            path = ROOT / f"data/parsed/{name}.parquet"
            old = pl.read_parquet(path)
            new = pl.DataFrame(rows, infer_schema_length=None).select(
                [pl.col(c).cast(t) for c, t in old.schema.items()])
            merged = pl.concat([old.filter(~pl.col("date").is_in(dates)), new]).sort("date", maintain_order=True)
            tmp = path.with_suffix(".tmp.parquet")
            merged.write_parquet(tmp)
            tmp.replace(path)
            print(name, "dates", len(dates), "rows", new.height, "total", merged.height, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
