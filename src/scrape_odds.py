"""boatrace.jp から3連単オッズページの生HTMLを収集する。

設計方針:
- **取得と解析を分離**。生HTMLを gzip で保存し、パースは別スクリプト。
  サイト構造変更やパーサのバグがあっても再取得不要。
- 対象レースは results.parquet（実際に開催されたレース）から生成
- 1req/秒、リトライ付き、再実行安全（取得済みスキップ）

URL: https://www.boatrace.jp/owpc/pc/race/odds3t?rno={r}&jcd={jcd:02d}&hd={yyyymmdd}

実行:  uv run python src/scrape_odds.py --start 20230701 --end 20250630
"""

import argparse
import gzip
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import requests


def log(msg: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "odds3t"
SLEEP_SEC = 1.0

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (research; boatrace-edge-model)"


def fetch(date: str, jcd: int, rno: int) -> str:
    dest = OUT / date[:6] / f"{date}_{jcd:02d}_{rno:02d}.html.gz"
    if dest.exists() and dest.stat().st_size > 200:
        return "skip"
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            time.sleep(SLEEP_SEC)
            if r.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(gzip.compress(r.content))
                return "ok"
            if r.status_code == 404:
                return "404"
            time.sleep(10 * (attempt + 1))  # 5xx等は間隔を空けて再試行
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    args = ap.parse_args()

    races = (
        pl.read_parquet(ROOT / "data" / "parsed" / "results.parquet")
        .select(["date", "jcd", "race_no"])
        .unique()
        .filter((pl.col("date") >= args.start) & (pl.col("date") <= args.end))
        .sort(["date", "jcd", "race_no"])
    )
    # 起動時に現在地を必ずログに残す（急停止→再開時にここを見れば分かる）
    todo = [
        (date, jcd, rno)
        for date, jcd, rno in races.iter_rows()
        if not (
            (d := OUT / date[:6] / f"{date}_{jcd:02d}_{rno:02d}.html.gz").exists()
            and d.stat().st_size > 200
        )
    ]
    log(f"START target={len(races)} done={len(races)-len(todo)} remaining={len(todo)}"
        f" (~{len(todo)*SLEEP_SEC/3600:.1f}h)")
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, (date, jcd, rno) in enumerate(todo):
        st = fetch(date, jcd, rno)
        counts[st] = counts.get(st, 0) + 1
        if st == "fail":
            log(f"FAIL {date} {jcd} {rno}")
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta_h = (len(todo) - i - 1) / rate / 3600 if rate > 0 else 0
            log(f"{i+1}/{len(todo)} {counts} 残り~{eta_h:.1f}h (直近位置 {date})")
    log(f"DONE {counts}")


if __name__ == "__main__":
    main()
