"""BoatraceCSV（2026-11-15 に公開停止）の代わりに、公開 JSON を日付範囲で取得する。

  openapi   https://boatraceopenapi.github.io/api/v1/YYYY/YYYYMMDD.json  出走表・直前情報（展示の進入・ST・展示タイム・
            チルト・部品交換）・結果。MIT、2026-01-01〜。当日分は約10分ごとに更新されるので、終わった日だけ取る
  turnmark  https://turnmark.github.io/api/v1/YYYY/YYYYMMDD.json       上に加えて確定オッズ（全7券種）。MIT、2026-01-01〜、
            前日分を翌朝に公開

どちらも非公式で、boatrace.jp から集めたもの。こちらからは boatrace.jp に触らない。1件/秒、取得済みは飛ばす、
429/503 で止まる。取得記録は data/raw/<source>/acquisition.jsonl。

実行: .venv/bin/python src/acquire_openapi.py --start 2026-11-13 --end 2026-11-21 [--source openapi turnmark]
"""
import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"openapi": "https://boatraceopenapi.github.io/api/v1/{y}/{d}.json",
           "turnmark": "https://turnmark.github.io/api/v1/{y}/{d}.json"}
SLEEP_SEC = 1.0


def check(content, day):
    """中身の最低限の検査: programs.stadiums.*.races.* の日付が全て day。"""
    data = json.loads(content)
    races = [r for s in data["programs"]["stadiums"].values() for r in s["races"].values()]
    if not races or any(r["date"] != f"{day:%Y-%m-%d}" for r in races):
        raise ValueError(f"Unexpected content for {day}")
    return len(races)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--source", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    today_jst = datetime.now(timezone(timedelta(hours=9))).date()
    if end >= today_jst:
        raise SystemExit("Only finished days (before today in JST) are fetched")
    session = requests.Session()
    counts = {}
    for source in args.source:
        out = ROOT / "data/raw" / source
        out.mkdir(parents=True, exist_ok=True)
        with (out / "acquisition.jsonl").open("a") as log:
            d = start
            while d <= end:
                path = out / f"{d:%Y%m%d}.json"
                if path.exists():
                    counts[(source, "skip")] = counts.get((source, "skip"), 0) + 1
                    d += timedelta(days=1)
                    continue
                url = SOURCES[source].format(y=f"{d:%Y}", d=f"{d:%Y%m%d}")
                r = session.get(url, timeout=60)
                if r.status_code in (429, 503):
                    raise RuntimeError(f"Stop on throttling: {r.status_code}, Retry-After={r.headers.get('Retry-After')}")
                races = 0
                if r.status_code == 200:
                    races = check(r.content, d)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(r.content)
                    tmp.replace(path)
                elif r.status_code != 404:
                    r.raise_for_status()
                log.write(json.dumps(dict(url=url, status=r.status_code, races=races,
                                          sha256=hashlib.sha256(r.content).hexdigest(),
                                          fetched_at=datetime.now(timezone.utc).isoformat())) + "\n")
                log.flush()
                counts[(source, r.status_code)] = counts.get((source, r.status_code), 0) + 1
                time.sleep(SLEEP_SEC)
                d += timedelta(days=1)
    print("done", {f"{s}:{k}": v for (s, k), v in counts.items()})


if __name__ == "__main__":
    main()
