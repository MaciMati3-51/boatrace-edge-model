"""odds3t 生HTML(gzip) → odds3t.parquet（レース×120組番のオッズ）。

テーブル構造（実物から解読、2023-07-01ページで確認）:
- <tbody class="is-p3-0"> 1個に 20行 × 6列 = 120 の oddsPoint セル（行優先順）
- 列 c (0..5) = 1着艇 c+1
- 行 R (0..19): t = R//4 → 2着艇: {1..6}−{1着} を昇順に並べた t 番目
               r = R%4  → 3着艇: {1..6}−{1着,2着} を昇順に並べた r 番目

欠場等でオッズが無い組は非数値（"欠場" 等）→ null で残す。
"""

import gzip
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "odds3t"
OUT = ROOT / "data" / "parsed"

TBODY = re.compile(r'<tbody class="is-p3-0">(.*?)</tbody>', re.S)
ODDS = re.compile(r'<td class="oddsPoint[^"]*">([^<]*)</td>')


def parse_html(h: str) -> list[dict] | None:
    m = TBODY.search(h)
    if not m:
        return None  # 中止・販売なしページ
    vals = ODDS.findall(m.group(1))
    if len(vals) != 120:
        return None
    rows: list[dict] = []
    for i, v in enumerate(vals):
        big_r, c = divmod(i, 6)
        t, r = divmod(big_r, 4)
        first = c + 1
        second_cands = [b for b in range(1, 7) if b != first]
        second = second_cands[t]
        third_cands = [b for b in range(1, 7) if b not in (first, second)]
        third = third_cands[r]
        try:
            odds = float(v.strip())
        except ValueError:
            odds = None
        rows.append({"combo": f"{first}-{second}-{third}", "odds": odds})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    n_bad = n_corrupt = 0
    for gz in sorted(SRC.rglob("*.html.gz")):
        date, jcd, rno = gz.stem.replace(".html", "").split("_")
        # 収集中に強制終了されるとサイズだけ確定してデータ未フラッシュのゼロ埋め
        # ファイルが残る（WSL2 で実際に発生）。スキップ判定はサイズしか見ないので
        # ここで弾く。該当ファイルは削除すれば次回収集時に取り直される。
        try:
            h = gzip.decompress(gz.read_bytes()).decode("utf-8", errors="replace")
        except (gzip.BadGzipFile, EOFError, OSError):
            print(f"CORRUPT {gz}")
            n_corrupt += 1
            continue
        rows = parse_html(h)
        if rows is None:
            n_bad += 1
            continue
        for row in rows:
            row.update({"date": date, "jcd": int(jcd), "race_no": int(rno)})
        all_rows += rows
    df = pl.DataFrame(all_rows)
    df.write_parquet(OUT / "odds3t.parquet")
    n_races = df.select(["date", "jcd", "race_no"]).unique().height if len(df) else 0
    print(f"races={n_races} rows={len(df)} bad_pages={n_bad} corrupt={n_corrupt}")


if __name__ == "__main__":
    main()
