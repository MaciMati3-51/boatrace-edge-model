"""LZH → TXT 一括解凍。

lhafile が C 拡張で Python 3.14 ヘッダ不在のためビルド不可
→ uv 管理の 3.12 で実行する（プロジェクト venv とは独立）:

  uv run --no-project --python 3.12 --with lhafile python src/extract_lzh.py
"""

from pathlib import Path

import lhafile

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SRC = ROOT / "lzh"
DST = ROOT / "txt"
# 本体は 2019-07 から。それより前の LZH（2016-07〜2019-06）は extract_lzh_ext.py が txt_ext/ へ展開する。
# ここで txt/ に出すと parse_b_meta.py などが本体に混ぜてしまう（2026-11-22 に一度起きた）
MAIN_START_YYMM = "201907"


def main() -> None:
    n_ok = n_skip = n_err = 0
    for lzh in sorted(SRC.rglob("*.lzh")):
        # data/raw/lzh/K/202606/k260615.lzh → data/raw/txt/K/202606/K260615.TXT
        rel = lzh.relative_to(SRC)
        if rel.parent.name < MAIN_START_YYMM:
            continue
        outdir = DST / rel.parent
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
