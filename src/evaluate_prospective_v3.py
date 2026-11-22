"""prereg/live_v3_design.md §4 の前向き検証。v2・v3 とも再学習・係数変更はしない。

1. prereg/live_v3_freeze.json のハッシュを照合する（v3 のモデル・係数・コード、v2 の凍結物）
2. 対象は 20260910〜。うち 20260923〜 は v3 の設計・凍結より後に初めて取得したデータ（主判定はここ）
3. v2 と v3 を同じレースで予測し、確定払戻・締切約9分前の板（od2）と照合する

実行: .venv/bin/python src/evaluate_prospective_v3.py
出力: prereg/live_v3_prospective_result.json（上書きしない）
"""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import polars as pl

import evaluate_prospective_v2 as ep2
from holdout import HOLDOUT_END
import live_market_v2 as m2
import live_model as v1
import live_v2 as v2
import live_v3 as v3

ROOT, KEY, BK = v3.ROOT, v3.KEY, v3.BK
START, FRESH = "20260910", "20260923"
OUT = ROOT / "prereg/live_v3_prospective_result.json"
FREEZE = ROOT / "prereg/live_v3_freeze.json"
NAMES = ["1着", "2連単", "2連複"]


def ci(diff, dates):
    return v2.day_bootstrap(diff if diff.ndim == 2 else diff[:, None], dates)


def run(start=START, fresh=FRESH, end=None, out=OUT, rehearsal=False):
    """rehearsal=True は配線の試運転（学習済みの期間で回すので数値に意味は無い。凍結照合を飛ばす）。"""
    def segments(dates):
        return {"all": np.ones(len(dates), bool), "seen_0910_0922": dates < fresh, "fresh": dates >= fresh}

    if not rehearsal:
        frozen = json.loads(FREEZE.read_text())["sha256"]
        for path, sha in frozen.items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
                raise SystemExit(f"Frozen artifact changed: {path}")
        if (start, fresh, end, out) != (START, FRESH, None, OUT):
            raise SystemExit("The registered evaluation takes no overrides")
    if out.exists():
        raise SystemExit("Result already recorded; refusing overwrite")
    if start <= HOLDOUT_END:
        raise SystemExit("Prospective period overlaps the sealed holdout")
    end = end or min(max(p.stem for p in (ROOT / "data/raw/boatracecsv/od2").glob("*.csv")),
                     pl.read_parquet(ROOT / "data/parsed/results.parquet")["date"].max())
    f3 = v3.load_frame(lo=start, hi=end)
    if f3["date"].max() != end:
        raise SystemExit("v3 feature cache does not reach the evaluation end; run v3_features.py")
    report = dict(period=[start, end], fresh_from=fresh, evaluated_at=datetime.now(timezone.utc).isoformat(),
                  frozen_hashes_verified=not rehearsal, rehearsal=rehearsal)

    # 展示タイムが締切前60分以内に出ていて、K文書の値と一致するレースだけ（v2 の前向き検証と同じ条件）
    frames = {k: ep2.raw_previews(k, start, end) for k in ("tkz", "od2")}
    in_window = ((pl.col("minutes_before_scheduled_deadline") > 0)
                 & (pl.col("minutes_before_scheduled_deadline") <= 60))
    tk = frames["tkz"].filter(in_window)
    tenji_pre = pl.concat([tk.select(*KEY, pl.lit(b).cast(pl.Int64).alias("boat"),
                                     pl.col(f"艇{b}_展示タイム").cast(pl.Float64, strict=False).alias("pre"))
                           for b in range(1, 7)])
    f2 = v3.v2_frame(end).filter(pl.col("date") >= start)
    pre_ok = (f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="left").group_by(KEY)
              .agg(ok=pl.col("pre").eq_missing(pl.col("tenji")).all(), n=pl.col("pre").is_not_null().sum())
              .filter(pl.col("ok") & (pl.col("n") == 6)).select(KEY))
    mismatch = f2.select(*BK, "tenji").join(tenji_pre, on=BK).filter(pl.col("pre") != pl.col("tenji")).height
    common = pre_ok.join(f3.select(KEY).unique(), on=KEY)
    f2, f3 = f2.join(common, on=KEY).sort(BK), f3.join(common, on=KEY).sort(BK)
    keys = f3.select(KEY).unique(maintain_order=True)
    if not f2.select(KEY).unique(maintain_order=True).equals(keys):
        raise SystemExit("v2 and v3 race sets differ")
    report["races"] = dict(predictable=keys.height, tenji_preview_vs_k_mismatched_boats=mismatch)

    p1_2, j2 = v2.load_model(v2.OUT / "model.pkl").predict(f2)
    p1_3, j3 = v3.load_model(v3.OUT / "model.pkl").predict(f3)
    l2, l3 = v1.losses(f2, p1_2, j2), v1.losses(f3, p1_3, j3)
    dates = keys["date"].to_numpy()
    order = np.argsort(f3["rank"].to_numpy().reshape(-1, 6), axis=1)
    market = v3.final_market_loss(keys, order)

    # Q1: v3 − v2 / Q2: 単体 − 確定市場
    report["Q1"], report["Q2"] = {}, {}
    for name, seg in segments(dates).items():
        if not seg.any():
            continue
        d = (l3 - l2)[seg]
        report["Q1"][name] = dict(races=int(seg.sum()), order=NAMES, v2=l2[seg].mean(axis=0).tolist(),
                                  v3=l3[seg].mean(axis=0).tolist(), v3_minus_v2=d.mean(axis=0).tolist(),
                                  day_bootstrap_95ci=ci(d, dates[seg]))
        q2 = {}
        for j, pool in ((1, "2連単"), (2, "2連複")):
            ok = seg & np.isfinite(market[:, j - 1])
            q2[pool] = dict(races=int(ok.sum()), final_market=float(market[ok, j - 1].mean()))
            for label, loss in (("v3", l3), ("v2", l2)):
                diff = loss[ok, j] - market[ok, j - 1]
                q2[pool][f"{label}_minus_final_market"] = dict(mean=float(diff.mean()),
                                                               day_bootstrap_95ci=ci(diff, dates[ok])[0])
        report["Q2"][name] = q2
    main = report["Q1"].get("fresh")
    if main:
        report["Q1"]["verdict"] = ("v3 better" if main["day_bootstrap_95ci"][1][1] < 0
                                   and main["day_bootstrap_95ci"][2][1] < 0 else "not established")
        report["Q2"]["verdict"] = {pool: ("v3 alone beats the final market"
                                          if q["v3_minus_final_market"]["day_bootstrap_95ci"][1] < 0
                                          else "not established") for pool, q in report["Q2"]["fresh"].items()}

    # Q3・Q4: 締切約9分前の板との合成、確定オッズ期待値での紙上回収率
    od = frames["od2"].filter(in_window)
    report["minutes_before_deadline_od2"] = np.quantile(
        od["minutes_before_scheduled_deadline"].to_numpy(), [.05, .5, .95]).tolist()
    coefs = {"v3": json.loads((v3.OUT / "market_coefficients.json").read_text()),
             "v2": json.loads((v2.OUT / "market_coefficients.json").read_text())}
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet").filter(pl.col("date") >= start)
    d = keys.with_row_index("i").join(od, on=KEY).sort("i")
    idx = d["i"].to_numpy()
    o2t_all, o2f_all = m2.board_arrays(d)
    first, second = order[idx, 0] + 1, order[idx, 1] + 1
    report["Q3"], report["Q4"] = {}, {}
    for pool in m2.POOLS:
        pay = (payouts.filter(pl.col("bet_type") == m2.LABELS[pool]).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        pay = d.select(KEY).join(pay, on=KEY, how="left", maintain_order="left")
        win = [(x, y) if pool == "2t" else (min(x, y), max(x, y)) for x, y in zip(first, second)]
        normal = ((pay["n"].to_numpy() == 1)
                  & np.array([c == f"{x}-{y}" for c, (x, y) in zip(pay["combo"].to_list(), win)]))
        complete = np.isfinite(o2t_all).all(axis=1) & (np.isfinite(o2f_all).all(axis=1) if pool == "2f" else True)
        keep = normal & complete
        o2t, o2f, rows = o2t_all[keep], o2f_all[keep], idx[keep]
        y = np.array([m2.POOLS[pool].index(w) for w, k in zip(win, keep) if k])
        amount = pay["amount"].to_numpy().astype(float)[keep]
        day = d["date"].to_numpy()[keep]
        board = o2t if pool == "2t" else o2f
        r = np.arange(len(y))
        final_market = -np.log(np.clip((100 / amount) / m2.TAKE[pool], 1e-15, 1))
        loss, ev = {}, {}
        for label, joint in (("v3", j3), ("v2", j2)):
            p = joint[rows] if pool == "2t" else m2.quinella_from_exacta(joint[rows])
            P = m2.distribution([np.log(np.clip(p, 1e-15, 1))] + m2.market_inputs(pool, o2t, o2f),
                                coefs[label][pool]["combined"])
            settle = m2.settle_factor(m2.drift_design(pool, P, o2t, o2f), coefs[label][pool]["drift"])
            loss[label] = -np.log(np.clip(P[r, y], 1e-15, 1))
            ev[label] = P * board * settle
        captured = -np.log(m2.normalize_inverse(board)[r, y])
        q3, q4 = {}, {}
        for name, seg in segments(day).items():
            if not seg.any():
                continue
            item = dict(races=int(seg.sum()), final_market=float(final_market[seg].mean()),
                        captured_market=float(captured[seg].mean()))
            for label in ("v3", "v2"):
                diff = (loss[label] - final_market)[seg]
                item[f"{label}_combined"] = float(loss[label][seg].mean())
                item[f"{label}_combined_minus_final_market"] = dict(mean=float(diff.mean()),
                                                                    day_bootstrap_95ci=ci(diff, day[seg])[0])
            diff = (loss["v3"] - loss["v2"])[seg]
            item["v3_combined_minus_v2_combined"] = dict(mean=float(diff.mean()),
                                                         day_bootstrap_95ci=ci(diff, day[seg])[0])
            q3[name] = item
            q4[name] = {label: {f"ev_settle_gt_{t}": m2.paper(ev[label][seg] > t, y[seg], amount[seg], day[seg])
                                for t in (1.0, 1.1, 1.2)} for label in ("v3", "v2")}
        report["Q3"][pool], report["Q4"][pool] = q3, q4
        print(pool, json.dumps(q3, ensure_ascii=False), flush=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("period", "races", "Q1", "Q2")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
