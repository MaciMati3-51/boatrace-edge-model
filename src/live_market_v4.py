"""v4 の市場合成・オッズ変動補正・紙上候補。計算は v2/v3 と同じで、確率だけ v4 に替える。

live_market_v2.py・live_market_v3.py は凍結対象なので書き換えず、関数を借りる。od2（締切約9分前の板）は
2連単・2連複だけなので、合成と期待値はこの2券種。3連単・3連複は v4 の確率をそのまま出す（板が無い）。

学習: .venv/bin/python src/live_market_v4.py train --tag live_v4 --oos A.parquet B.parquet --lo 20260719 --hi 20261112
      （live_v4.py train の後。--oos は時系列外の予測。学習に使った日の予測が混じっていれば止まる）
予測: .venv/bin/python src/live_market_v4.py predict --input race.csv --odds odds.csv --out prereg/live_v4_YYYYMMDD_JJ_RR
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

import evaluate_prospective_v2 as ep2
import live_market_v2 as m2
import live_market_v3 as m3
import live_v4 as fund

ROOT, KEY = fund.ROOT, fund.KEY
POOLS, LABELS = m2.POOLS, m2.LABELS


def boards(lo, hi):
    """boatracecsv の od2（生CSV）。締切前60分以内に撮られた板だけ。"""
    od = ep2.raw_previews("od2", lo, hi)
    return od.filter((pl.col("minutes_before_scheduled_deadline") > 0)
                     & (pl.col("minutes_before_scheduled_deadline") <= 60))


def train(tag, oos_paths, lo, hi):
    """時系列外の予測（oos_paths）× 板 × 払戻で、合成係数と変動補正を推定して <tag>/ に保存する。"""
    out = ROOT / "data/models" / tag
    meta = json.loads((out / "metadata.json").read_text())
    oos = pl.concat([pl.read_parquet(p) for p in oos_paths]).filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
    if (oos["model_trained_before"] > oos["date"]).any():
        raise ValueError("OOS predictions include in-sample rows")
    if oos.select(KEY).unique().height != oos.height:
        raise ValueError("OOS files overlap")
    tmp = out / "_oos_window.parquet"
    oos.sort(KEY).write_parquet(tmp)
    try:
        data = m3.load_dataset(tmp, boards(lo, hi))
    finally:
        tmp.unlink()
    saved = {}
    for pool, d in data.items():
        saved[pool] = m3.fit_pool(pool, d, np.ones(len(d["y"]), bool))
        saved[pool]["counts"] = d["counts"]
        print(pool, json.dumps(saved[pool], ensure_ascii=False), flush=True)
    saved.update(fundamental_model_sha256=meta["model_sha256"], fitted_on=[lo, hi],
                 oos_sha256={str(Path(p).resolve().relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                             for p in oos_paths},
                 generated_at=datetime.now(timezone.utc).isoformat(),
                 note="Coefficients fit on out-of-sample live_v4 predictions and boards captured about 9 minutes "
                      "before the deadline.")
    (out / "market_coefficients.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    return saved


def apply(pool, coef, p_2t, o2t, o2f):
    """確率と板 → (最終確率, 撮影オッズでの期待値, 確定オッズでの期待値)。p_2t は (レース, 30)。"""
    p = p_2t if pool == "2t" else m2.quinella_from_exacta(p_2t)
    market = m2.market_inputs(pool, o2t, o2f)
    final = m2.distribution([np.log(np.clip(p, 1e-15, 1))] + market, coef[pool]["combined"])
    board = o2t if pool == "2t" else o2f
    ev_captured = final * board
    ev_settle = ev_captured * m2.settle_factor(m2.drift_design(pool, final, o2t, o2f), coef[pool]["drift"])
    return p, market, final, ev_captured, ev_settle


def predict(input_path, odds_path, out, tag="live_v4"):
    model_dir = ROOT / "data/models" / tag
    inputs = fund.v2.validate_input(pl.read_csv(input_path, schema_overrides={"date": pl.String}))
    odds = m2.validate_odds(pl.read_csv(odds_path, schema_overrides={"date": pl.String}))
    input_keys = inputs.select(KEY + ["deadline_at"]).unique()
    if odds.select(KEY + ["deadline_at"]).unique().join(input_keys, on=KEY + ["deadline_at"], how="anti").height:
        raise ValueError("Odds race/deadline does not match exhibition input")
    coef = json.loads((model_dir / "market_coefficients.json").read_text())
    sha = hashlib.sha256((model_dir / "model.pkl").read_bytes()).hexdigest()
    if sha != coef["fundamental_model_sha256"]:
        raise ValueError("Fundamental model changed; recalibrate market coefficients")
    if out.exists():
        raise ValueError("Output directory already exists")
    table = {}
    for group in odds.partition_by(KEY + ["pool"]):
        row = group.row(0, named=True)
        lookup = {(a, b): o for a, b, o in zip(group["first"], group["second"], group["odds"])}
        table[(row["date"], row["jcd"], row["race_no"], row["pool"])] = (
            np.array([lookup[pair] for pair in POOLS[row["pool"]]], dtype=float), row)
    race_keys = inputs.select(KEY).unique().sort(KEY)
    for key in race_keys.iter_rows():
        if (*key, "2t") not in table:
            raise ValueError("2連単 board is required (the 2連複 market also uses it)")
    out.mkdir(parents=True)
    # fundamental.csv は全券種（1着・2連単・2連複・3連単・3連複）の確率。オッズ不使用
    # 予測の行はレースキーの昇順（live_frame が BK で並べる）＝ race_keys と同じ順
    _, joint, _ = fund.predict(input_path, out / "fundamental.csv", model_dir / "model.pkl")
    if len(joint) != race_keys.height:
        raise ValueError("Prediction rows do not match the input races")
    records = []
    for i, key in enumerate(race_keys.iter_rows()):
        o2t = table[(*key, "2t")][0][None]
        for pool in POOLS:
            if (*key, pool) not in table:
                continue
            o2f = table[(*key, "2f")][0][None] if pool == "2f" else np.full((1, 15), np.nan)
            p, market, final, ev_captured, ev_settle = apply(pool, coef, joint[i:i + 1], o2t, o2f)
            board = o2t if pool == "2t" else o2f
            meta = table[(*key, pool)][1]
            for j, (a, b) in enumerate(POOLS[pool]):
                records.append(dict(date=key[0], jcd=key[1], race_no=key[2], pool=pool, first=a, second=b,
                                    odds=float(board[0, j]), captured_at=meta["captured_at"],
                                    deadline_at=meta["deadline_at"], p_fundamental=float(p[0, j]),
                                    p_market=float(np.exp(market[0][0, j])),
                                    p_market_from_2t=float(np.exp(market[1][0, j])) if pool == "2f" else None,
                                    p_final=float(final[0, j]), ev_captured=float(ev_captured[0, j]),
                                    ev_settle=float(ev_settle[0, j]),
                                    paper_candidate=bool(ev_settle[0, j] > 1)))
    pl.DataFrame(records).write_csv(out / "final.csv")
    (out / "metadata.json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model=tag, model_sha256=sha,
        calibration=coef, odds_sha256=hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        note="paper_candidate uses ev_settle (expected settled odds). Research output; profitability not established."),
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("--tag", default="live_v4")
    ap.add_argument("--oos", type=Path, nargs="+")
    ap.add_argument("--lo")
    ap.add_argument("--hi")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--odds", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "train":
        if not (args.oos and args.lo and args.hi):
            ap.error("train requires --oos, --lo and --hi")
        train(args.tag, args.oos, args.lo, args.hi)
    elif args.input and args.odds and args.out:
        predict(args.input, args.odds, args.out, args.tag)
    else:
        ap.error("predict requires --input, --odds and --out")
