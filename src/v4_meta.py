"""v4 のレース情報: 進入固定の旗と、表記ゆれを直したレース種別。

v3（parse_b_meta.py → race_meta.parquet → v3_features.py）には欠けが2つある。v3 は凍結なので直さず、
v4 が別の列として足す。

1. 進入固定: v3 はレース名の先頭の語だけを読むので、「進入固定」がレース名に入る場合（進入固定戦隊 など、
   年約175件）しか拾えない。実際は見出しの別の欄に書かれていて年約1,400件（全レースの約2.5%）ある

       　５Ｒ  予　選　　　 進入固定 Ｈ１８００ｍ  電話投票締切予定１４：５３

2. レース種別: 2020年までの表記「予　選」「一　般」（全角空白入り）を v3 は「予」「一」と読み、不明(0)に
   落としていた（2019〜2020 の 54〜65% が不明）。空白を詰めてから読む

出力: data/parsed/race_meta_v4.parquet  1行 = (date, jcd, race_no)、列 fixed4（0/1）・rtype4（0〜5）
実行: .venv/bin/python src/v4_meta.py        （B の TXT を取り込んだ後に流す）
"""
import re
from pathlib import Path

import polars as pl

from parse_b_meta import norm

ROOT = Path(__file__).resolve().parents[1]
SRC = [ROOT / "data/raw/txt/B", ROOT / "data/raw/txt_ext/B"]  # 本体と、2019-07 より前（extract_lzh_ext.py）
OUT = ROOT / "data/parsed/race_meta_v4.parquet"
KEY = ["date", "jcd", "race_no"]
# 「　５Ｒ  <レース名> [進入固定] Ｈ１８００ｍ」。レース名に空白が入ることがあるので距離の欄までを丸ごと取る
RACE = re.compile(r"^[\s　]*([０-９0-9]{1,2})[ＲR][\s　]+(.*?)[ＨH][０-９0-9]{3,4}[ｍm]")
FIXED = "進入固定"


def race_type(name):
    """v3 の RACE_TYPES と同じ段階（5 優勝戦 / 4 準優勝戦 / 3 特賞・特選・選抜 / 2 予選 / 1 一般 / 0 その他）。"""
    if "準優" in name:
        return 4
    if "優勝戦" in name:
        return 5
    if any(k in name for k in ("特賞", "特選", "選抜")):
        return 3
    if "予選" in name or name.endswith("予"):  # 「めざまし戦予」は欄の幅で切れた「…予選」
        return 2
    if name in ("一般", "一般戦", "一"):
        return 1
    return 0


def parse_file(path):
    date = "20" + path.stem[1:7]
    rows, jcd = [], None
    for line in path.read_bytes().decode("cp932", errors="replace").split("\r\n"):
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd = int(m.group(1))
            continue
        if re.match(r"^\d{2}BEND", line):
            jcd = None
            continue
        if jcd is None:
            continue
        m = RACE.match(line)
        if m:
            text = norm(m.group(2))
            fixed = FIXED in text
            name = text[:-len(FIXED)] if text.endswith(FIXED) and len(text) > len(FIXED) else text
            rows.append(dict(date=date, jcd=jcd, race_no=int(norm(m.group(1))), fixed4=int(fixed),
                             rtype4=race_type(name), race_name=name))
    return rows


def build(files=None):
    files = sorted(f for src in SRC for f in src.glob("*/*.TXT")) if files is None else files
    rows = [r for f in files for r in parse_file(f)]
    return pl.DataFrame(rows).unique(subset=KEY, keep="first").sort(KEY)


def main():
    df = build()
    df.write_parquet(OUT)
    print(f"{OUT}: {df.height:,} races ({df['date'].min()}〜{df['date'].max()}), "
          f"fixed {df['fixed4'].mean():.4f}, unknown type {(df['rtype4'] == 0).mean():.4f}")


if __name__ == "__main__":
    main()
