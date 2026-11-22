"""acquire_openapi.py で取った JSON を表にする。

  previews(lo, hi, source)  直前情報（スタート展示の進入・展示ST・展示タイム・チルト・部品交換の数）。
                            v5_entry.exhibition() と同じ列（KEY + c1..c6 + s1..s6）に、t1..t6（展示タイム）などを足す
  final_odds(lo, hi)        turnmark の確定オッズ（2連単30・2連複15・3連単120・3連複20）。組番は1始まりの文字列

過去日の preview は取り直した後の最終値で、撮影時刻は無い（boatraceopenapi の schema.md「時点（リーク）に関する注意」）。
"""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
KEY = ["date", "jcd", "race_no"]
BOATS = range(1, 7)


def _races(source, lo, hi):
    for path in sorted((ROOT / "data/raw" / source).glob("*.json")):
        if lo <= path.stem <= hi:
            data = json.loads(path.read_bytes())
            for stadium in data["programs"]["stadiums"].values():
                yield from stadium["races"].values()


def previews(lo, hi, source="openapi"):
    rows = []
    for race in _races(source, lo, hi):
        pv = race.get("preview") or {}
        racers = {int(k): v for k, v in (pv.get("racers") or {}).items()}
        row = dict(date=race["date"].replace("-", ""), jcd=int(race["stadium_number"]), race_no=int(race["race_number"]),
                   closed_at=race.get("closed_at"))
        for b in BOATS:
            r = racers.get(b, {})
            row[f"c{b}"] = r.get("course_number")
            row[f"s{b}"] = r.get("start_timing")
            row[f"t{b}"] = r.get("exhibition_time")
            row[f"tilt{b}"] = r.get("tilt_adjustment")
            row[f"parts{b}"] = len(r.get("parts") or []) if r else None
        rows.append(row)
    schema = {"date": pl.String, "jcd": pl.Int64, "race_no": pl.Int64, "closed_at": pl.String,
              **{f"{k}{b}": (pl.Int64 if k in ("c", "parts") else pl.Float64)
                 for b in BOATS for k in ("c", "s", "t", "tilt", "parts")}}
    df = pl.DataFrame(rows, schema=schema)
    if df.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate races in previews")
    return df.sort(KEY)


POOL_KEYS = {"2t": "exacta", "2f": "quinella", "3t": "trifecta", "3f": "trio"}


def final_odds(lo, hi):
    """縦持ち: KEY + pool + combo（"1-2" など）+ odds。欠場艇を含む組は null。"""
    rows = []
    for race in _races("turnmark", lo, hi):
        odds = race.get("odds")
        if not odds:
            continue
        key = (race["date"].replace("-", ""), int(race["stadium_number"]), int(race["race_number"]))
        for pool, name in POOL_KEYS.items():
            stack = [((), odds.get(name) or {})]
            while stack:
                path, node = stack.pop()
                if isinstance(node, dict):
                    stack.extend(((*path, k), v) for k, v in node.items())
                else:
                    rows.append((*key, pool, "-".join(path), None if node is None else float(node)))
    return pl.DataFrame(rows, schema={"date": pl.String, "jcd": pl.Int64, "race_no": pl.Int64, "pool": pl.String,
                                      "combo": pl.String, "odds": pl.Float64}, orient="row").sort([*KEY, "pool", "combo"])
