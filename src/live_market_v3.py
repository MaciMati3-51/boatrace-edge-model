"""v3 の市場合成・オッズ変動補正・紙上候補。計算は v2（live_market_v2.py）と同じで、確率だけ v3 に替える。

live_market_v2.py は前向き検証の凍結対象（ハッシュ照合される）なので書き換えず、関数を借りる。

学習: .venv/bin/python src/live_market_v3.py train   （live_v3.py train の後）
予測: .venv/bin/python src/live_market_v3.py predict --input race.csv --odds odds.csv --out prereg/live_v3_YYYYMMDD_JJ_RR
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

import live_market_v2 as m2
import live_v3 as fund

ROOT, KEY, OUT = fund.ROOT, fund.KEY, fund.OUT
POOLS, LABELS = m2.POOLS, m2.LABELS
CALIBRATION, TEST = m2.CALIBRATION, m2.TEST


def load_dataset(oos_path=None, od=None, payouts=None):
    """v3 の時系列外予測 × od2 板 × 通常払戻。締切前60分以内の板、正常6艇完走だけ。"""
    if oos_path is None:
        oos_path = OUT / "oos_predictions.parquet"
        meta = json.loads((OUT / "metadata.json").read_text())
        if hashlib.sha256(oos_path.read_bytes()).hexdigest() != meta["oos_sha256"]:
            raise ValueError("OOS predictions changed after live_v3 training")
    oos = pl.read_parquet(oos_path)
    if od is None:
        od = pl.read_parquet(ROOT / "data/parsed/boatracecsv/od2.parquet")
    od = od.filter((pl.col("minutes_before_scheduled_deadline") > 0)
                   & (pl.col("minutes_before_scheduled_deadline") <= 60))
    if od.unique(KEY).height != od.height:
        raise ValueError("Duplicate market records")
    d = oos.join(od, on=KEY, how="inner").sort(KEY)
    o2t, o2f = m2.board_arrays(d)
    joint = np.array(d["joint"].to_list())
    rank = np.array(d["rank"].to_list())
    first, second = rank.argmin(axis=1) + 1, np.argsort(rank, axis=1)[:, 1] + 1
    if payouts is None:
        payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet")
    payouts = payouts.filter(pl.col("date") >= d["date"].min())
    data = {}
    for pool in POOLS:
        pay = (payouts.filter(pl.col("bet_type") == LABELS[pool]).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        pay = d.select(KEY).join(pay, on=KEY, how="left", maintain_order="left")
        winners = [(a, b) if pool == "2t" else (min(a, b), max(a, b)) for a, b in zip(first, second)]
        normal = ((pay["n"].to_numpy() == 1)
                  & np.array([c == f"{a}-{b}" for c, (a, b) in zip(pay["combo"].to_list(), winners)])
                  & (np.nan_to_num(pay["amount"].to_numpy().astype(float)) > 0))
        complete = np.isfinite(o2t).all(axis=1) & (np.isfinite(o2f).all(axis=1) if pool == "2f" else True)
        keep = normal & complete
        p = joint if pool == "2t" else m2.quinella_from_exacta(joint)
        data[pool] = dict(dates=d["date"].to_numpy()[keep], keys=d.select(KEY).filter(pl.Series(keep)),
                          p=p[keep], o2t=o2t[keep], o2f=o2f[keep],
                          y=np.array([POOLS[pool].index(w) for w in winners])[keep],
                          amount=pay["amount"].to_numpy().astype(float)[keep],
                          counts=dict(boards=int(d.height), no_normal_payout=int((~normal).sum()),
                                      incomplete_board=int((normal & ~complete).sum()), used=int(keep.sum())))
    return data


def fit_pool(pool, d, mask):
    """合成係数とオッズ変動補正を mask のレースで推定する。"""
    market = m2.market_inputs(pool, d["o2t"][mask], d["o2f"][mask])
    features = [np.log(np.clip(d["p"][mask], 1e-15, 1))] + market
    coef = m2.fit(features, d["y"][mask])
    combined = m2.distribution(features, coef)
    board = (d["o2t"] if pool == "2t" else d["o2f"])[mask]
    drift = m2.fit_drift(m2.drift_design(pool, combined, d["o2t"][mask], d["o2f"][mask]),
                         board, d["y"][mask], d["amount"][mask])
    return dict(combined=coef, drift=drift, races=int(mask.sum()),
                inputs=["log p_fundamental"] + (["log q_2t"] if pool == "2t" else ["log q_2f", "log q_2f_from_2t"]))


def train():
    data = load_dataset()
    report = dict(calibration_period=list(CALIBRATION), test_period=list(TEST),
                  status="exploratory historical comparison; normal six-finisher races; boards captured about 9 "
                         "minutes before the deadline. The same period was used to develop live_v1 and live_v2.",
                  final_market="probability of the winning combination from the settled payout: (100/payout)/TAKE",
                  bet_rules={"ev_captured": "P_final x captured odds > 1",
                             "ev_settle": "P_final x captured odds x E[settled/captured | win] > 1"},
                  pools={})
    saved = {}
    for pool, d in data.items():
        calib = (d["dates"] >= CALIBRATION[0]) & (d["dates"] <= CALIBRATION[1])
        test = (d["dates"] >= TEST[0]) & (d["dates"] <= TEST[1])
        report["pools"][pool] = m2.evaluate(pool, d, calib, test)
        t = report["pools"][pool]["methods"]["test"]
        print(pool, json.dumps({k: (v["logloss"], v["minus_final_market"]["mean"]) for k, v in t.items()
                                if isinstance(v, dict)}, ensure_ascii=False), flush=True)
        saved[pool] = fit_pool(pool, d, calib | test)  # 保存用: 全期間（7/19〜9/9）で再推定
    saved.update(fundamental_model_sha256=json.loads((OUT / "metadata.json").read_text())["model_sha256"],
                 fitted_on=[CALIBRATION[0], TEST[1]], generated_at=datetime.now(timezone.utc).isoformat(),
                 note="Coefficients fit on live_v3 out-of-sample predictions; the saved fundamental model is "
                      "refit through 20260909.")
    (OUT / "market_coefficients.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "prereg/live_market_v3_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def predict(input_path, odds_path, out):
    inputs = fund.v2.validate_input(pl.read_csv(input_path, schema_overrides={"date": pl.String}))
    odds = m2.validate_odds(pl.read_csv(odds_path, schema_overrides={"date": pl.String}))
    input_keys = inputs.select(KEY + ["deadline_at"]).unique()
    if odds.select(KEY + ["deadline_at"]).unique().join(input_keys, on=KEY + ["deadline_at"], how="anti").height:
        raise ValueError("Odds race/deadline does not match exhibition input")
    coef = json.loads((OUT / "market_coefficients.json").read_text())
    sha = hashlib.sha256((OUT / "model.pkl").read_bytes()).hexdigest()
    if sha != coef["fundamental_model_sha256"]:
        raise ValueError("Fundamental model changed; recalibrate market coefficients")
    if out.exists():
        raise ValueError("Output directory already exists")
    boards = {}
    for group in odds.partition_by(KEY + ["pool"]):
        row = group.row(0, named=True)
        lookup = {(a, b): o for a, b, o in zip(group["first"], group["second"], group["odds"])}
        boards[(row["date"], row["jcd"], row["race_no"], row["pool"])] = (
            np.array([lookup[pair] for pair in POOLS[row["pool"]]], dtype=float), row)
    for key in inputs.select(KEY).unique().iter_rows():
        if (*key, "2t") not in boards:
            raise ValueError("2連単 board is required (the 2連複 market also uses it)")
    out.mkdir(parents=True)
    fund.predict(input_path, out / "fundamental.csv")
    p = pl.read_csv(out / "fundamental.csv", schema_overrides={"date": pl.String})
    records = []
    for key in inputs.select(KEY).unique().sort(KEY).iter_rows():
        pp = p.filter(pl.all_horizontal([pl.col(c) == v for c, v in zip(KEY, key)])).sort("first", "second")
        p2t = pp["p_2t"].to_numpy()[None]
        o2t, _ = boards[(*key, "2t")]
        for pool in POOLS:
            if (*key, pool) not in boards:
                continue
            o2f = boards[(*key, "2f")][0][None] if pool == "2f" else np.full((1, 15), np.nan)
            p_f = p2t if pool == "2t" else m2.quinella_from_exacta(p2t)
            market = m2.market_inputs(pool, o2t[None], o2f)
            final = m2.distribution([np.log(np.clip(p_f, 1e-15, 1))] + market, coef[pool]["combined"])
            board = o2t[None] if pool == "2t" else o2f
            ev_captured = final * board
            ev_settle = ev_captured * m2.settle_factor(m2.drift_design(pool, final, o2t[None], o2f),
                                                       coef[pool]["drift"])
            meta = boards[(*key, pool)][1]
            for j, (a, b) in enumerate(POOLS[pool]):
                records.append(dict(date=key[0], jcd=key[1], race_no=key[2], pool=pool, first=a, second=b,
                                    odds=float(board[0, j]), captured_at=meta["captured_at"],
                                    deadline_at=meta["deadline_at"], p_fundamental=float(p_f[0, j]),
                                    p_market=float(np.exp(market[0][0, j])),
                                    p_market_from_2t=float(np.exp(market[1][0, j])) if pool == "2f" else None,
                                    p_final=float(final[0, j]), ev_captured=float(ev_captured[0, j]),
                                    ev_settle=float(ev_settle[0, j]),
                                    paper_candidate=bool(ev_settle[0, j] > 1)))
    pl.DataFrame(records).write_csv(out / "final.csv")
    (out / "metadata.json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v3", model_sha256=sha,
        calibration=coef, odds_sha256=hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        note="paper_candidate uses ev_settle (expected settled odds). Research output; profitability not established."),
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
