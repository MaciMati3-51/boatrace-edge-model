"""v5 の市場合成・オッズ変動補正・紙上候補。v4 の確率に展示進入の補正（v5_entry.py）を掛け、v4 と同じ式で合成する。

v4 のファイル（live_v4.py・live_market_v4.py、data/models/live_v4/）は書き換えない。v4 の凍結モデルの出力に
data/models/live_v5/entry_coefficients.json の補正を掛ける。設計と事前登録は prereg/live_v5_design.md（不採用）・prereg/live_v5b_design.md（凍結した版）。

学習: .venv/bin/python src/live_market_v5.py train      （v5b_course.py fit の後。07-19〜11-12 の時系列外予測で推定）
予測: .venv/bin/python src/live_market_v5.py predict --input race.csv --odds odds.csv --out prereg/live_v5_YYYYMMDD_JJ_RR
      入力CSVの exhibition_course・exhibition_st を使う（空欄・並べ替えでなければ補正なし＝v4 と同じ確率）
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

import live_market_v4 as m4
import live_v4 as f4
import v5_entry as v5
import v5b_course as vb

ROOT, KEY = v5.ROOT, v5.KEY
m2, m3 = m4.m2, m4.m3
POOLS = m2.POOLS
V4_MODEL = ROOT / "data/models/live_v4/model.pkl"
ENTRY = v5.OUT / "entry_coefficients.json"
MARKET = v5.OUT / "market_coefficients.json"


def train():
    entry = json.loads(ENTRY.read_text())
    keys, p1, joint, tri, order = v5.load_oos(v5.EVAL_OOS, *v5.EVAL_WINDOW)
    course, st = v5.arrays(keys, v5.exhibition(*v5.EVAL_WINDOW))
    _, qj, _ = v5.correct(p1, joint, tri, course, st, entry, vb.extra_for(entry, keys, course))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v5_oos.parquet"
        keys.with_columns(joint=pl.Series(qj.tolist()), rank=pl.Series(np.argsort(order, axis=1).tolist())
                          ).write_parquet(path)
        data = m3.load_dataset(path, m4.boards(*v5.EVAL_WINDOW))
    saved = {}
    for pool, d in data.items():
        saved[pool] = m3.fit_pool(pool, d, np.ones(len(d["y"]), bool))
        saved[pool]["counts"] = d["counts"]
        print(pool, json.dumps(saved[pool], ensure_ascii=False), flush=True)
    saved.update(fundamental_model_sha256=v5.sha(V4_MODEL), entry_coefficients_sha256=v5.sha(ENTRY),
                 fitted_on=list(v5.EVAL_WINDOW), generated_at=datetime.now(timezone.utc).isoformat(),
                 note="Coefficients fit on out-of-sample live_v4a predictions corrected by the v5 exhibition-entry "
                      "layer, and boards captured about 9 minutes before the deadline.")
    MARKET.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    return saved


def predict(input_path, odds_path, out):
    inputs = f4.v2.validate_input(pl.read_csv(input_path, schema_overrides={"date": pl.String}))
    odds = m2.validate_odds(pl.read_csv(odds_path, schema_overrides={"date": pl.String}))
    input_keys = inputs.select(KEY + ["deadline_at"]).unique()
    if odds.select(KEY + ["deadline_at"]).unique().join(input_keys, on=KEY + ["deadline_at"], how="anti").height:
        raise ValueError("Odds race/deadline does not match exhibition input")
    entry, coef = json.loads(ENTRY.read_text()), json.loads(MARKET.read_text())
    sha = v5.sha(V4_MODEL)
    if sha != coef["fundamental_model_sha256"] or sha != entry["v4_model_sha256"]:
        raise ValueError("v4 model changed; refit the v5 coefficients")
    if v5.sha(ENTRY) != coef["entry_coefficients_sha256"]:
        raise ValueError("Entry coefficients changed; refit the market coefficients")
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
    p1, joint, tri = f4.predict(input_path, out / "fundamental_v4.csv", V4_MODEL)
    if len(joint) != race_keys.height:
        raise ValueError("Prediction rows do not match the input races")
    course, st = v5.arrays(race_keys, v5.from_inputs(inputs))
    boats = race_keys.with_row_index("order").join(inputs.select(*KEY, "boat", "racer_id"), on=KEY).sort(
        "order", "boat").drop("order")
    history = pl.read_parquet(ROOT / "data/parsed/results.parquet",
                              columns=["date", "racer_id", "entry_course", "rank_raw"]).filter(
        pl.col("date") < inputs["date"].min())
    p1, joint, tri = v5.correct(p1, joint, tri, course, st, entry, vb.extra_for(entry, race_keys, course, boats, history))
    f4.probability_rows(race_keys, p1, joint, tri).write_csv(out / "fundamental.csv")
    records = []
    for i, key in enumerate(race_keys.iter_rows()):
        o2t = table[(*key, "2t")][0][None]
        for pool in POOLS:
            if (*key, pool) not in table:
                continue
            o2f = table[(*key, "2f")][0][None] if pool == "2f" else np.full((1, 15), np.nan)
            p, market, final, ev_captured, ev_settle = m4.apply(pool, coef, joint[i:i + 1], o2t, o2f)
            board = o2t if pool == "2t" else o2f
            meta = table[(*key, pool)][1]
            for j, (a, b) in enumerate(POOLS[pool]):
                records.append(dict(date=key[0], jcd=key[1], race_no=key[2], pool=pool, first=a, second=b,
                                    odds=float(board[0, j]), captured_at=meta["captured_at"],
                                    deadline_at=meta["deadline_at"], exhibition_changed=bool(
                                        (course[i] != v5.BOATS).any()),
                                    p_fundamental=float(p[0, j]), p_market=float(np.exp(market[0][0, j])),
                                    p_market_from_2t=float(np.exp(market[1][0, j])) if pool == "2f" else None,
                                    p_final=float(final[0, j]), ev_captured=float(ev_captured[0, j]),
                                    ev_settle=float(ev_settle[0, j]), paper_candidate=bool(ev_settle[0, j] > 1)))
    pl.DataFrame(records).write_csv(out / "final.csv")
    (out / "metadata.json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v5", model_sha256=sha,
        entry_coefficients=entry, calibration=coef,
        odds_sha256=hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        note="v4 probabilities corrected by exhibition entry (v5_entry.py). paper_candidate uses ev_settle. "
             "Research output; profitability not established."), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("--input", type=Path)
    ap.add_argument("--odds", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "train":
        train()
    elif args.input and args.odds and args.out:
        predict(args.input, args.odds, args.out)
    else:
        ap.error("predict requires --input, --odds and --out")
