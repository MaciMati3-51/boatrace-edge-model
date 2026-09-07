"""mbrace の番組表(B)・競走成績(K) LZH を日次一括ダウンロードする。

URLパターン（実物で確認済み 2026-07-19）:
  https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh  … 競走成績
  https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh  … 番組表

- 1req/秒のレート制限（公営サーバーへの配慮）
- 取得済みファイルはスキップ（再実行安全）
- 404 は「開催なし/未公開」として記録のみ
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://www1.mbrace.or.jp/od2"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "lzh"
SLEEP_SEC = 1.0

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (research; contact via boatrace-edge-model)"


def fetch_day(d: date, kind: str) -> str:
    """kind: 'K' or 'B'。戻り値はステータス文字列。"""
    ym = d.strftime("%Y%m")
    fname = f"{kind.lower()}{d.strftime('%y%m%d')}.lzh"
    out = OUT / kind / ym
    out.mkdir(parents=True, exist_ok=True)
    dest = out / fname
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    url = f"{BASE}/{kind}/{ym}/{fname}"
    r = session.get(url, timeout=30)
    time.sleep(SLEEP_SEC)
    if r.status_code == 404:
        return "404"
    r.raise_for_status()
    # LZHヘッダ簡易検証（3バイト目以降に -lhX- が入る）
    if b"-lh" not in r.content[:8]:
        return "not-lzh"
    dest.write_bytes(r.content)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    d = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    counts: dict[str, int] = {}
    while d <= end:
        for kind in ("B", "K"):
            st = fetch_day(d, kind)
            counts[st] = counts.get(st, 0) + 1
            if st not in ("ok", "skip"):
                print(f"{d} {kind}: {st}", flush=True)
        if d.day == 1:
            print(f"-- {d} まで到達 {counts}", flush=True)
        d += timedelta(days=1)
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
