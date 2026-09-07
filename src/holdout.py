"""封印ホールドアウトの境界を1か所に固定し、コードで守らせる。

README §5/§8 は「封印ホールドアウト=直近1年、開封は1回のみ、開封回数を記録」と
決めているが、**2026-09-07 まで境界の日付はどこにも書かれておらず、コードは
一度も封印を守っていなかった**。`fundamental_wf.parquet` は 20260718 まで予測を
持ち、G1 も全期間で報告されていた。この モジュールはその穴を塞ぐためにある。

## 境界

    HOLDOUT_START = "20250719"

決め方: 封印を宣言した時点（2026-09-07）のデータ末尾 20260718 から1年遡った日。
**この日付は二度と動かさない。** 動かせば「良く出るほうを選ぶ」ができてしまい、
封印の意味が消える。データが増えても境界は据え置き、増えた分は封印側に積む。

## walk-forward 予測を封印側にも作ること自体は違反ではない

walk-forward は過去だけで学習するので因果的に汚れていない。封印が防ぐのは
**「集計値を何度も見て、良かったものを選ぶ」** ことであって、予測の生成ではない。
よってここで止めるのは *評価* であって *予測* ではない。開封時には封印側の
予測が必要になるので、`train_fundamental.py` は今まで通り全期間を予測する。

## 使い方

    from holdout import HOLDOUT_START, walk_forward, unseal

    df = walk_forward(df)              # 既定はこれ。封印側を落とす
    if args.unseal:
        unseal("理由", __file__)       # UNSEAL_LOG.md に記録してから触る

`unseal()` は理由の記録を強制する。理由なしでは通らない。
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_START = "20250719"
UNSEAL_LOG = ROOT / "UNSEAL_LOG.md"


def walk_forward(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """封印側を落として walk-forward 側だけ返す。解析スクリプトの既定経路。"""
    return df.filter(pl.col(date_col) < HOLDOUT_START)


def sealed(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """封印側だけ返す。`unseal()` を通してから使うこと。"""
    return df.filter(pl.col(date_col) >= HOLDOUT_START)


def describe(df: pl.DataFrame, date_col: str = "date", key: list[str] | None = None) -> None:
    key = key or ["date", "jcd", "race_no"]
    u = df.select(key).unique()
    w = u.filter(pl.col(date_col) < HOLDOUT_START).height
    s = u.filter(pl.col(date_col) >= HOLDOUT_START).height
    print(f"封印境界 {HOLDOUT_START}: walk-forward {w:,}レース / 封印 {s:,}レース")


def unseal(reason: str, script: str) -> None:
    """封印側を見る。理由を UNSEAL_LOG.md に追記してからでないと通さない。"""
    reason = (reason or "").strip()
    if len(reason) < 10:
        raise SystemExit(
            "開封には理由が要る（10文字以上）。README §8: 各回の日付・対象・結果を記録する。"
        )
    stamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M %z")
    if not UNSEAL_LOG.exists():
        UNSEAL_LOG.write_text(
            "# 封印ホールドアウト 開封記録\n\n"
            f"境界: `{HOLDOUT_START}` 以降（`src/holdout.py` で固定）。\n"
            "README §8 は開封1回のみと定めている。ここに残らない開封は無効。\n\n"
        )
    with UNSEAL_LOG.open("a") as f:
        f.write(f"- **{stamp}** `{Path(script).name}` — {reason}\n")
    print(f"⚠ 封印を開けた。{UNSEAL_LOG.name} に記録した: {reason}")
