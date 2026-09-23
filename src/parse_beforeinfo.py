"""Parse saved official beforeinfo HTML; never read race results or fetch URLs.

Usage: .venv/bin/python src/parse_beforeinfo.py
Strict six-boat parser: incomplete/changed pages fail rather than become zeroes.
"""
import html
import re
import unicodedata
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
KEY = ["date", "jcd", "race_no", "boat"]


def clean(value):
    return unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", "", value))).strip()


def parse(page, date, jcd, race_no):
    rows = {}
    for body in re.findall(r"<tbody\b[^>]*>(.*?)</tbody>", page, re.S):
        first = re.search(r"<tr\b[^>]*>(.*?)</tr>", body, re.S)
        if not first or not re.search(r'is-boatColor[1-6] is-fs14', first[1]):
            continue
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", first[1], re.S)
        if len(cells) != 10:
            raise ValueError("Unexpected beforeinfo columns")
        boat = int(clean(cells[0]))
        racer = re.search(r"toban=(\d+)", cells[1])
        if boat in rows or not racer:
            raise ValueError("Duplicate boat or missing racer")
        parts = [clean(x) for x in re.findall(r"<li\b[^>]*>(.*?)</li>", cells[7], re.S)]
        rows[boat] = dict(date=date, jcd=jcd, race_no=race_no, boat=boat,
                          racer_id=int(racer[1]), tenji=float(clean(cells[4])),
                          tilt=float(clean(cells[5])),
                          parts_raw="|".join(parts), propeller_raw=clean(cells[6]))
    starts = re.findall(
        r'<span class="table1_boatImage1Number[^\"]*">([1-6])</span>.*?'
        r'<span class="table1_boatImage1Time[^\"]*">([^<]*)</span>', page, re.S)
    if set(rows) != set(range(1, 7)) or len(starts) != 6 or len({b for b, _ in starts}) != 6:
        raise ValueError("Incomplete boat/start exhibition data")
    for course, (boat, raw) in enumerate(starts, 1):
        raw = clean(raw)
        if not re.fullmatch(r"F?\.?\d+\.?(?:\d+)?", raw):
            raise ValueError(f"Unsupported exhibition ST: {raw!r}")
        rows[int(boat)].update(exhibition_course=course, exhibition_st_raw=raw,
                              exhibition_st=(-1 if raw.startswith("F") else 1)*float(raw.removeprefix("F")))
    return [rows[b] for b in sorted(rows)]


def main():
    records = []
    for path in sorted((ROOT / "data/raw/beforeinfo").glob("*/*.html")):
        date, jcd, race = path.stem.split("_")
        records.extend(dict(row, source_path=str(path.relative_to(ROOT)))
                       for row in parse(path.read_text(), date, int(jcd), int(race)))
    if not records:
        raise SystemExit("No saved beforeinfo HTML")
    df = pl.DataFrame(records)
    if df.unique(KEY).height != df.height:
        raise ValueError("Duplicate race/boat keys")
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").select(KEY + ["racer_id"])
    if df.join(entries, on=KEY + ["racer_id"], how="anti").height:
        raise ValueError("Race/boat/racer does not match existing entries")
    out = ROOT / "data/parsed/beforeinfo.parquet"
    df.write_parquet(out)
    print(f"Saved {df.height} matched boat records to {out}")


if __name__ == "__main__":
    main()
