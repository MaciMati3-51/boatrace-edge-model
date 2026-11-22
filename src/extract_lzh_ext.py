"""2019-07 より前の LZH を、本体とは別の data/raw/txt_ext/ に解凍する。

本体の data/raw/txt/ に置くと、parse_kb.py（全 TXT から entries/results を作り直す）が古い年まで
本体の parquet に取り込んでしまい、v3 の凍結系列の入力が変わる。置き場を分けて既存のスクリプトから
見えないようにする。取り込みは build_history_ext.py。

実行: uv run --no-project --python 3.12 --with lhafile python src/extract_lzh_ext.py
"""
from pathlib import Path

import lhafile

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SRC, DST = ROOT / "lzh", ROOT / "txt_ext"
CUT = "201907"  # 本体の先頭月。これより前の月だけ


def main():
    n_ok = n_skip = n_err = 0
    for lzh in sorted(SRC.glob("*/*/*.lzh")):
        if lzh.parent.name >= CUT:
            continue
        outdir = DST / lzh.parent.parent.name / lzh.parent.name
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            f = lhafile.Lhafile(str(lzh))
            for info in f.infolist():
                dest = outdir / info.filename
                if dest.exists() and dest.stat().st_size == info.file_size:
                    n_skip += 1
                    continue
                dest.write_bytes(f.read(info.filename))
                n_ok += 1
        except Exception as e:  # 壊れたアーカイブは記録して続行
            print(f"ERR {lzh}: {e}")
            n_err += 1
    print(f"extracted={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
