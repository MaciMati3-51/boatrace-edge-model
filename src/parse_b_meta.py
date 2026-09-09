"""B文書のヘッダから節タイトル・レース種別・開催時間帯を取り出す。

`parse_kb.py` の `parse_b_file` は出走表の行しか読んでおらず、**どの節か・どの
時間帯かが分からない**（2026-09-09 に必要になって追加）。番組表の見出しには

    ボートレース大　村   　７月　１日  日刊スポーツ杯　　　  第　２日
    　１Ｒ  予選　　　　          Ｈ１８００ｍ  電話投票締切予定１５：１５

の2つが入っている。ここから

  - 節タイトル … オールレディース/ヴィーナスシリーズ/ルーキーシリーズ/マスターズ の判定に使う
                 （G1・SG は表記が無いので A1 の人数を代理指標にする）
  - レース種別 … 予選 / 準優勝戦 / 優勝戦 / 一般 / 選抜 …
  - 締切時刻   … 節の最終レースの締切から モーニング/デイ/ナイター/ミッドナイト を決める

出力: data/parsed/race_meta.parquet  1行 = (date, jcd, race_no)
実行: uv run python src/parse_b_meta.py
"""

import re
import unicodedata
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "txt" / "B"
OUT = ROOT / "data" / "parsed" / "race_meta.parquet"

TITLE = re.compile(r"^ボートレース(.{1,8}?)\s{2,}.*?日\s+(.*?)\s+第\s*[０-９\d]+日\s*$")
RACE = re.compile(r"^[\s　]*([０-９0-9]{1,2})[ＲR]\s+(\S+)")
DEADLINE = re.compile(r"電話投票締切予定([０-９0-9]{1,2}[：:][０-９0-9]{2})")
SERIES = ["オールレディース", "ヴィーナス", "ルーキー", "マスターズ"]


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "").replace("　", "")


def series_of(title: str) -> str:
    for k in SERIES:
        if k in title:
            return k
    return "一般"


def parse_file(path: Path) -> list[dict]:
    date = "20" + path.stem[1:7]
    rows: list[dict] = []
    jcd = title = None
    for line in path.read_bytes().decode("cp932", errors="replace").split("\r\n"):
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd, title = int(m.group(1)), None
            continue
        if re.match(r"^\d{2}BEND", line):
            jcd = None
            continue
        if jcd is None:
            continue
        if title is None and line.startswith("ボートレース"):
            t = TITLE.match(line)
            title = norm(t.group(2)) if t else ""
            continue
        m = RACE.match(line)
        if m and ("Ｈ" in line or "H" in line):
            d = DEADLINE.search(line)
            rows.append({
                "date": date,
                "jcd": jcd,
                "race_no": int(norm(m.group(1))),
                "title": title or "",
                "series": series_of(title or ""),
                "race_type": norm(m.group(2)),
                "deadline": norm(d.group(1)) if d else None,
            })
    return rows


def main() -> None:
    files = sorted(SRC.glob("*/*.TXT"))
    rows = [r for f in files for r in parse_file(f)]
    df = pl.DataFrame(rows).unique(subset=["date", "jcd", "race_no"], keep="first")
    mins = (
        pl.col("deadline").str.slice(0, 2).cast(pl.Int64) * 60
        + pl.col("deadline").str.slice(3, 2).cast(pl.Int64)
    )
    # ミッドナイトは 0時台に食い込むので、5時未満は翌日扱いで +24h する
    df = df.with_columns(min_of_day=pl.when(mins < 300).then(mins + 1440).otherwise(mins))
    last = df.group_by(["date", "jcd"]).agg(
        last_min=pl.col("min_of_day").max(), first_min=pl.col("min_of_day").min()
    )
    df = df.join(last, on=["date", "jcd"]).with_columns(
        slot=pl.when(pl.col("last_min") >= 22 * 60).then(pl.lit("ミッドナイト"))
        .when(pl.col("last_min") >= 20 * 60).then(pl.lit("ナイター"))
        .when(pl.col("last_min") >= 17 * 60 + 30).then(pl.lit("サマータイム"))
        .when(pl.col("first_min") <= 10 * 60).then(pl.lit("モーニング"))
        .otherwise(pl.lit("デイ"))
    )
    df.write_parquet(OUT)
    print(f"保存: {OUT}  {df.height:,}レース ({df['date'].min()}〜{df['date'].max()})")
    print(df.group_by("slot").agg(n=pl.len(), 最終R締切=pl.col("last_min").median()).sort("n", descending=True))
    print(df.group_by("series").len().sort("len", descending=True))
    print(df.group_by("race_type").len().sort("len", descending=True).head(10))


if __name__ == "__main__":
    main()
