"""手動保存した公式2連単オッズ画像を30通りの表へ変換する。

実行例:
  uv run --no-project --with rapidocr-onnxruntime python src/parse_manual_2tf.py \
      picture/74tukkii --output data/parsed/odds2tf_manual.parquet

画像の表はPC版公式画面（1084px幅）を前提とする。OCR結果は全30通りが
揃い、数値・信頼度ともに妥当なものだけを書き出す。画像は原本として残す。
"""

import argparse
import re
from pathlib import Path

import cv2
import polars as pl
from rapidocr_onnxruntime import RapidOCR


ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^(?P<date>\d{8})_(?P<jcd>\d{2})_(?P<race>\d{2})_2rentan\.png$")
NUMBER = re.compile(r"^\d+(?:\.\d)?$")
# OCRが壊した1セルを原画像で目視照合した補正。原本は picture/ に保持される。
TEXT_FIXES = {("20260908_09_04_2rentan.png", "'66"): "99.7"}


def odds_grid(result, row_gap: float, image_name: str) -> list[list[tuple[float, float, str, float]]]:
    """OCR結果から、小数で表示される2連単オッズを5×6表へ並べる。"""
    values: list[tuple[float, float, str, float]] = []
    for box, text, confidence in result or []:
        text = TEXT_FIXES.get((image_name, text), text)
        x = sum(point[0] for point in box) / 4
        y = sum(point[1] for point in box) / 4
        # ほぼ全て小数表記。高配当は整数表示（例: 1054）になるため3桁以上も許容。
        if NUMBER.fullmatch(text) and ("." in text or len(text) >= 3):
            values.append((y, x, text, confidence))
    grid: list[list[tuple[float, float, str, float]]] = []
    for value in sorted(values):
        if not grid or value[0] - grid[-1][0][0] > row_gap:
            grid.append([value])
        else:
            grid[-1].append(value)
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--output", type=Path, default=ROOT / "data/parsed/odds2tf_manual.parquet")
    args = ap.parse_args()

    ocr = RapidOCR()
    rows: list[dict] = []
    rejected: list[str] = []
    for image in sorted(args.image_dir.glob("*.png")):
        m = NAME.match(image.name)
        if not m:
            rejected.append(f"{image.name}: 命名規則外")
            continue
        # 表だけに切り詰める。ヘッダ全体をOCRすると不要な日本語認識に時間を使う。
        table = cv2.imread(str(image))[830:, :]
        result, _ = ocr(table)
        grid = odds_grid(result, 12, image.name)
        if len(grid) != 5 or any(len(row) != 6 for row in grid):
            # 端の1セルだけを見落とした画像は2倍拡大で再認識する。
            enlarged = cv2.resize(table, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            result, _ = ocr(enlarged)
            grid = odds_grid(result, 24, image.name)
        if len(grid) != 5 or any(len(row) != 6 for row in grid):
            rejected.append(f"{image.name}: OCR行={','.join(str(len(row)) for row in grid)}")
            continue
        for second_row, row in enumerate(grid):
            for first, (_, _, text, confidence) in enumerate(sorted(row, key=lambda v: v[1]), start=1):
                second = [boat for boat in range(1, 7) if boat != first][second_row]
                rows.append(
                    {
                        "date": m["date"],
                        "jcd": int(m["jcd"]),
                        "race_no": int(m["race"]),
                        "first": first,
                        "second": second,
                        "combo": f"{first}-{second}",
                        "odds": float(text),
                        "ocr_confidence": float(confidence),
                        "image": str(image.resolve().relative_to(ROOT)),
                    }
                )

    out = pl.DataFrame(rows).sort(["date", "jcd", "race_no", "first", "second"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)
    print(f"accepted={out.select(['date','jcd','race_no']).unique().height} races rows={out.height}")
    if rejected:
        print("REJECTED (原本を確認):")
        print("\n".join(rejected))


if __name__ == "__main__":
    main()
