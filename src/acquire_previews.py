"""boatracecsv（GitHub Pages の静的CSV）から直前情報を日付範囲で取得する。

acquire_previews_2026.py は期間が固定だったので、範囲を引数にした版。取得済みは飛ばす。
429/503 が出たら止める。boatrace.jp には触らない。

実行: .venv/bin/python src/acquire_previews.py --start 2026-09-23 --end 2026-11-12
"""
import argparse
import csv
import hashlib
import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/boatracecsv"
KINDS = ("tkz", "stt", "sui", "od2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    session = requests.Session()
    d, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    counts = {}
    with (OUT / "acquisition.jsonl").open("a") as log:
        while d <= end:
            for kind in KINDS:
                path = OUT / kind / f"{d:%Y%m%d}.csv"
                if path.exists():
                    counts["skip"] = counts.get("skip", 0) + 1
                    continue
                url = f"https://boatracecsv.github.io/data/previews/{kind}/{d:%Y/%m/%d}.csv"
                r = session.get(url, timeout=30)
                if r.status_code in (429, 503):
                    raise RuntimeError(f"Stop on throttling: {r.status_code}, Retry-After={r.headers.get('Retry-After')}")
                rows = 0
                if r.status_code == 200:
                    records = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
                    if not records or "レースコード" not in records[0] or any(
                            x["レースコード"][:8] != f"{d:%Y%m%d}" for x in records):
                        raise ValueError(f"Unexpected content: {url}")
                    rows = len(records)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(r.content)
                    tmp.replace(path)
                elif r.status_code != 404:
                    r.raise_for_status()
                log.write(json.dumps(dict(url=url, status=r.status_code, rows=rows,
                                          sha256=hashlib.sha256(r.content).hexdigest(),
                                          fetched_at=datetime.now(timezone.utc).isoformat()),
                                     ensure_ascii=False) + "\n")
                log.flush()
                counts[r.status_code] = counts.get(r.status_code, 0) + 1
                time.sleep(0.5)
            d += timedelta(days=1)
    print("done", counts)


if __name__ == "__main__":
    main()
