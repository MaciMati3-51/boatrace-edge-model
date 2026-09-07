"""B(番組表)/K(競走成績) TXT → Parquet 3テーブル。

固定幅 + Shift-JIS(cp932) 全角混在のため、文字数でなく**バイト位置**で切る。
（全角=2バイト。文字位置ベースだと選手名の全角/半角混在で崩れる）

出力:
  data/parsed/entries.parquet  … 番組表（出走情報、締切時刻含む）
  data/parsed/results.parquet  … レース結果（着順・進入・ST・決まり手・気象）
  data/parsed/payouts.parquet  … 払戻（全券種、人気順位付き）

会場コード(jcd)はファイル内の NNKBGN / NNBBGN 区切りから取る。
"""

import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / "data" / "raw" / "txt"
OUT = ROOT / "data" / "parsed"

ZEN2HAN = str.maketrans("０１２３４５６７８９：．", "0123456789:.")


def z2h(s: str) -> str:
    return s.translate(ZEN2HAN)


def b2s(b: bytes) -> str:
    """バイト列を cp932 でデコードし前後空白（全角含む）を落とす。"""
    return b.decode("cp932", errors="replace").replace("　", " ").strip()


def fnum(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def inum(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


# ---------- K（競走成績） ----------

K_RACE_HEADER = re.compile(r"^\s{0,4}(\d{1,2})R\s")
# 例: "   1R  カタメン１予  H1800m  晴  風  北  3m  波  3cm"（空白可変）
K_METEO = re.compile(
    r"H(?P<dist>\d{3,4})m\s+(?P<weather>\S+)\s+風\s+(?P<wdir>\S+)\s+(?P<wspd>\d+)m"
    r"\s+波\s+(?P<wave>\d+)cm"
)
PAYOUT_TYPES = ("単勝", "複勝", "２連単", "２連複", "拡連複", "３連単", "３連複")


def parse_k_file(path: Path, date: str) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    payouts: list[dict] = []
    raw = path.read_bytes().split(b"\r\n")
    jcd = None
    race_no = None
    meteo: dict = {}
    kimarite = None
    cur_type = None  # 払戻の券種は複数行に跨る（拡連複・複勝）

    for bline in raw:
        line = bline.decode("cp932", errors="replace")
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd = int(m.group(1))
            continue
        if re.match(r"^\d{2}KEND", line):
            jcd = None
            continue
        if jcd is None:
            continue

        m = K_RACE_HEADER.match(line)
        if m and ("H" in line and "m" in line):
            race_no = int(m.group(1))
            meteo = {}
            mm = K_METEO.search(line.replace("　", " "))
            if mm:
                meteo = {
                    "distance": int(mm["dist"]),
                    "weather": mm["weather"],
                    "wind_dir": mm["wdir"],
                    "wind_speed": int(mm["wspd"]),
                    "wave": int(mm["wave"]),
                }
            kimarite = None
            cur_type = None
            continue

        # 列ヘッダ行の末尾に決まり手が乗る: "... ﾚｰｽﾀｲﾑ 逃げ"
        if "ﾚｰｽﾀｲﾑ" in line:
            tail = line.split("ﾚｰｽﾀｲﾑ")[-1].replace("　", " ").strip()
            kimarite = tail or None
            continue

        # 結果行: "  01  1 4433 名前(16bytes) 67  104  6.86   1    0.21     1.49.8"
        # 着順トークンは 01-06 / 00 / S0-S2 / F / L0-L1 / K0-K1 と多形
        # → 接頭辞の列挙でなく「トークン + 艇番 + 登番4桁」の構造でマッチ
        if race_no is not None and re.match(r"^  \S{1,2}\s+[1-6] \d{4} ", line):
            b = bline
            tail = b[29:]
            results.append(
                {
                    "date": date,
                    "jcd": jcd,
                    "race_no": race_no,
                    **meteo,
                    "kimarite": kimarite,
                    "rank_raw": b2s(b[0:4]),
                    "boat": inum(b2s(b[4:7])),
                    "racer_id": inum(b2s(b[7:12])),
                    "motor_no": inum(b2s(tail[0:4])),
                    "boat_no": inum(b2s(tail[4:8])),
                    "tenji": fnum(b2s(tail[8:14])),
                    "entry_course": inum(b2s(tail[14:18])),
                    "st": b2s(tail[18:28]),
                    "racetime": b2s(tail[28:40]),
                }
            )
            continue

        # 払戻行（券種名は先頭にある行と、続き行がある）
        if race_no is not None:
            s = line.replace("　", " ").strip()
            typ = next((t for t in PAYOUT_TYPES if s.startswith(t)), None)
            if typ:
                cur_type = typ
                s = s[len(typ):].strip()
            elif cur_type is None or not s:
                continue
            # "1-5-6     3800  人気    12" / "1          100" / 続き行 "5-6 1080 人気 11"
            pm = re.match(r"^([\d\-=]+)\s+(\d+)(?:\s+人気\s+(\d+))?", s)
            if pm and cur_type:
                payouts.append(
                    {
                        "date": date,
                        "jcd": jcd,
                        "race_no": race_no,
                        "bet_type": cur_type,
                        "combo": pm.group(1),
                        "amount": int(pm.group(2)),
                        "ninki": inum(pm.group(3) or ""),
                    }
                )
            elif not pm:
                cur_type = None  # 特払い・不成立などは券種を閉じる
    return results, payouts


# ---------- B（番組表） ----------

B_RACE_HEADER = re.compile(r"^[\s　]*([０-９0-9]{1,2})[ＲR]")
B_DEADLINE = re.compile(r"電話投票締切予定([０-９0-9]{1,2}[：:][０-９0-9]{2})")


def parse_b_file(path: Path, date: str) -> list[dict]:
    entries: list[dict] = []
    raw = path.read_bytes().split(b"\r\n")
    jcd = None
    race_no = None
    deadline = None
    for bline in raw:
        line = bline.decode("cp932", errors="replace")
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd = int(m.group(1))
            continue
        if re.match(r"^\d{2}BEND", line):
            jcd = None
            continue
        if jcd is None:
            continue
        m = B_RACE_HEADER.match(line)
        if m and ("Ｈ" in line or "H" in line):
            race_no = int(z2h(m.group(1)))
            dm = B_DEADLINE.search(line)
            deadline = z2h(dm.group(1)) if dm else None
            continue
        # 出走行: "1 4433川野芽唯40福岡46A1 6.85 51.67 6.31 45.28 67 35.59104 28.30 ..."
        if race_no is not None and re.match(r"^[1-6] \d{4}", line):
            b = bline
            entries.append(
                {
                    "date": date,
                    "jcd": jcd,
                    "race_no": race_no,
                    "deadline": deadline,
                    "boat": inum(b2s(b[0:1])),
                    "racer_id": inum(b2s(b[2:6])),
                    "name": b2s(b[6:14]),
                    "age": inum(b2s(b[14:16])),
                    "branch": b2s(b[16:20]),
                    "weight": inum(b2s(b[20:22])),
                    "class": b2s(b[22:24]),
                    "nat_wr": fnum(b2s(b[24:29])),
                    "nat_2r": fnum(b2s(b[29:35])),
                    "loc_wr": fnum(b2s(b[35:40])),
                    "loc_2r": fnum(b2s(b[40:46])),
                    "motor_no": inum(b2s(b[46:49])),
                    "motor_2r": fnum(b2s(b[49:55])),
                    "boat_no": inum(b2s(b[55:58])),
                    "boat_2r": fnum(b2s(b[58:64])),
                    "setsu_seiseki": b2s(b[64:76]),
                }
            )
    return entries


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    all_payouts: list[dict] = []
    all_entries: list[dict] = []

    for txt in sorted(TXT.rglob("K*.TXT")):
        date = "20" + txt.stem[1:]  # K260615 → 20260615
        r, p = parse_k_file(txt, date)
        all_results += r
        all_payouts += p
    for txt in sorted(TXT.rglob("B*.TXT")):
        date = "20" + txt.stem[1:]
        all_entries += parse_b_file(txt, date)

    pl.DataFrame(all_results).write_parquet(OUT / "results.parquet")
    pl.DataFrame(all_payouts).write_parquet(OUT / "payouts.parquet")
    pl.DataFrame(all_entries).write_parquet(OUT / "entries.parquet")
    print(
        f"results={len(all_results)} payouts={len(all_payouts)} entries={len(all_entries)}"
    )


if __name__ == "__main__":
    main()
