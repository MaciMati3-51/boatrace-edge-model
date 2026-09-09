"""指定日のレースを、締切前に手に入る情報だけで予測して凍結する。

## なぜ専用スクリプトが要るか

`build_features.py` は entries と results を **inner join** している（開催済みレースだけ）。
未来の日付には K文書が無いので1行も残らない。また `tenji`（展示タイム）は K文書由来で、
**レース後にしか手に入らない**。よってここでは

  - 特徴量を entries + 過去の results だけから組む（tenji 系を全部落とす）
  - 学習も同じ縮小特徴量で行う（学習と予測の特徴量を必ず揃える）

展示タイムを落とすコストは実測 +0.0158（1着 log-loss 1.1848 → 1.2006、45,272レース）。
直前情報ページを手動で取れば取り戻せるが、それは別経路。

## 出力

    prereg/pred_YYYYMMDD.parquet (+ .csv)     1行 = (jcd, race_no, first, second)
        p_1st        … 1着確率（レース内で和1）
        p_2t         … 2連単 P(1着=first, 2着=second)、30通りで和1
        p_2f         … 2連複 P({first,second} が1-2着)、first<second の15行のみ非null

**事前登録用。** `data/` は .gitignore なので、出力は追跡される `prereg/` に置く。
予測はレース前に生成してコミットし、後から差し替えない。判定指標も同じディレクトリの
`prereg/YYYYMMDD.md` に、結果を見る前に書いておくこと。

実行: uv run python src/predict_day.py --date 20260910
      uv run python src/predict_day.py --date 20260909 --backtest   # 検算（結果と突合）
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from holdout import HOLDOUT_END, HOLDOUT_START

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "jcd", "race_no"]
N_BOATS = 6

# 締切前に確定している列だけ。tenji 系は K文書由来なので入れない。
FEATURES = [
    "boat", "jcd", "class_ord", "age", "weight",
    "nat_wr", "nat_2r", "loc_wr", "loc_2r", "motor_2r", "boat_2r",
    "hist_n", "hist_win_rate", "hist_top2_rate", "hist_mean_st", "hist_f_rate",
    "vb_win_rate",
]
PER_BOAT = [c for c in FEATURES if c not in ("boat", "jcd")]
# 2着モデルで「先着艇の状態」として渡す列
CONTEXT = ["class_ord", "nat_wr", "nat_2r", "loc_wr", "loc_2r",
           "motor_2r", "boat_2r", "hist_win_rate", "hist_top2_rate",
           "hist_mean_st", "hist_f_rate", "vb_win_rate"]
CLASS_ORD = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
PAIRS = np.array([(a, b) for a in range(N_BOATS) for b in range(N_BOATS) if a != b])


def history_as_of(results: pl.DataFrame, date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """その日より前だけを使った選手ローリング統計と会場×枠の1着率。"""
    past = results.filter(pl.col("date") < date)
    st = pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64)
    racer = (
        past.with_columns(st_num=st)
        .group_by("racer_id")
        .agg(
            hist_n=pl.len(),
            hist_win_rate=(pl.col("rank_raw") == "01").mean(),
            hist_top2_rate=pl.col("rank_raw").is_in(["01", "02"]).mean(),
            hist_mean_st=pl.col("st_num").mean(),
            hist_f_rate=pl.col("rank_raw").str.starts_with("F").mean(),
        )
    )
    vb = (
        past.group_by(["jcd", "boat"])
        .agg(vb_win_rate=(pl.col("rank_raw") == "01").mean(), vb_n=pl.len())
    )
    return racer, vb


def build_target(date: str) -> pl.DataFrame:
    """対象日の出走表 + 過去だけから作った特徴量。"""
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").filter(pl.col("date") == date)
    if entries.is_empty():
        raise SystemExit(f"{date} の番組表が entries.parquet に無い。download_lzh → extract → parse_kb の順に流す")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    racer, vb = history_as_of(results, date)
    df = (
        entries.join(racer, on="racer_id", how="left")
        .join(vb, on=["jcd", "boat"], how="left")
        .with_columns(class_ord=pl.col("class").replace_strict(CLASS_ORD, default=None))
        .sort(KEY + ["boat"])
    )
    full = df.group_by(KEY).len().filter(pl.col("len") == N_BOATS).select(KEY)
    return df.join(full, on=KEY).sort(KEY + ["boat"])


def stage1_matrix(df: pl.DataFrame) -> np.ndarray:
    n = df.height // N_BOATS
    jcd = df["jcd"].to_numpy().astype(float).reshape(n, N_BOATS)[:, :1]
    return np.hstack([jcd, df.select(PER_BOAT).to_numpy().astype(float).reshape(n, -1)])


def pair_matrix(x: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """(候補艇の全特徴) + (先着艇の状態) + (差分)。x は (レース,6,特徴)。"""
    rows = np.arange(len(first))
    ci = [FEATURES.index(c) for c in CONTEXT]
    cand, prev = x[rows, second], x[rows, first]
    return np.column_stack([cand, prev[:, ci], cand[:, ci] - prev[:, ci]])


def model(cat: list[int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=50, categorical_features=cat, random_state=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="予測する日 YYYYMMDD")
    ap.add_argument("--backtest", action="store_true", help="結果と突合して log-loss を出す")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    date = args.date
    if HOLDOUT_START <= date <= HOLDOUT_END:
        raise SystemExit(
            f"{date} は封印期間（{HOLDOUT_START}〜{HOLDOUT_END}）。unseal() を通さずには触らない"
        )

    # ---- 学習データ: 対象日より前の完走6艇レース ----
    base = pl.read_parquet(ROOT / "data/features/base.parquet").filter(pl.col("date") < date)
    ok = (
        base.group_by(KEY)
        .agg(n=pl.len(), c=pl.col("y_rank").is_not_null().all(), w=pl.col("y_win").sum())
        .filter((pl.col("n") == N_BOATS) & pl.col("c") & (pl.col("w") == 1))
        .select(KEY)
    )
    train = base.join(ok, on=KEY).sort(KEY + ["boat"])
    n_tr = train.height // N_BOATS
    x_tr = train.select(FEATURES).to_numpy().astype(float).reshape(n_tr, N_BOATS, -1)
    rank_tr = train["y_rank"].to_numpy().astype(np.int8).reshape(n_tr, N_BOATS)
    y1 = rank_tr.argmin(axis=1)
    print(f"学習 {n_tr:,}レース（〜{train['date'].max()}）/ 特徴量 {len(FEATURES)}列（展示タイム抜き）")

    # ---- 対象日 ----
    tgt = build_target(date)
    n = tgt.height // N_BOATS
    keys = tgt.select(KEY).unique(maintain_order=True)
    x_te = tgt.select(FEATURES).to_numpy().astype(float).reshape(n, N_BOATS, -1)
    print(f"予測 {n:,}レース / {tgt['jcd'].n_unique()}会場")

    # ---- stage1: レース単位6クラス softmax ----
    clf1 = model([0]).fit(stage1_matrix(train), y1)
    p1 = clf1.predict_proba(stage1_matrix(tgt))

    # ---- stage2: 1着を所与とした2着（残り5艇で正規化）----
    order = np.argsort(rank_tr, axis=1)
    keep = rank_tr >= 2
    race, cand = np.nonzero(keep)
    z2 = pair_matrix(x_tr[race], order[race, 0], cand)
    y2 = rank_tr[race, cand] == 2
    clf2 = model([FEATURES.index("boat"), FEATURES.index("jcd")]).fit(z2, y2)

    rr = np.repeat(np.arange(n), len(PAIRS))
    s2 = clf2.predict_proba(
        pair_matrix(x_te[rr], np.tile(PAIRS[:, 0], n), np.tile(PAIRS[:, 1], n))
    )[:, 1].reshape(n, len(PAIRS))
    p2 = np.zeros_like(s2)
    for a in range(N_BOATS):
        m = PAIRS[:, 0] == a
        p2[:, m] = s2[:, m] / np.clip(s2[:, m].sum(axis=1, keepdims=True), 1e-12, None)
    joint = p1[:, PAIRS[:, 0]] * p2                      # 2連単 30通り
    joint = joint / joint.sum(axis=1, keepdims=True)

    rows = []
    for i, k in enumerate(keys.iter_rows(named=True)):
        pf = {}
        for j, (a, b) in enumerate(PAIRS):
            pf[(min(a, b), max(a, b))] = pf.get((min(a, b), max(a, b)), 0.0) + joint[i, j]
        for j, (a, b) in enumerate(PAIRS):
            rows.append({
                **k, "first": a + 1, "second": b + 1,
                "p_1st": float(p1[i, a]), "p_2t": float(joint[i, j]),
                "p_2f": float(pf[(a, b)]) if a < b else None,
            })
    out = pl.DataFrame(rows)
    path = args.out or ROOT / "prereg" / f"pred_{date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    out.write_csv(path.with_suffix(".csv"))
    print(f"保存: {path}  {out.height:,}行（{n:,}レース×30）")

    if args.backtest:
        res = pl.read_parquet(ROOT / "data/parsed/results.parquet").filter(pl.col("date") == date)
        w = (
            res.group_by(KEY)
            .agg(w1=pl.col("boat").filter(pl.col("rank_raw") == "01").first(),
                 w2=pl.col("boat").filter(pl.col("rank_raw") == "02").first())
            .drop_nulls()
        )
        j = out.join(w, on=KEY).filter((pl.col("first") == pl.col("w1")) & (pl.col("second") == pl.col("w2")))
        ll1 = -np.log(np.clip(j["p_1st"].to_numpy(), 1e-12, None))
        ll2 = -np.log(np.clip(j["p_2t"].to_numpy(), 1e-12, None))
        f = out.join(w, on=KEY).filter(
            (pl.col("first") == pl.min_horizontal("w1", "w2")) & (pl.col("second") == pl.max_horizontal("w1", "w2"))
        )
        llf = -np.log(np.clip(f["p_2f"].to_numpy(), 1e-12, None))
        ci = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        print(f"\n===== backtest {date}  {j.height}レース =====")
        print(f"  1着   log-loss {ll1.mean():.4f} ±{ci(ll1):.4f}  (一様 {np.log(6):.4f})")
        print(f"  2連単 log-loss {ll2.mean():.4f} ±{ci(ll2):.4f}  (一様 {np.log(30):.4f})")
        print(f"  2連複 log-loss {llf.mean():.4f} ±{ci(llf):.4f}  (一様 {np.log(15):.4f})")


if __name__ == "__main__":
    main()
