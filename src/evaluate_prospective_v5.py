"""prereg/live_v5b_design.md §4 の前向き検証（1回だけ）。v4・v5 とも再学習・係数変更はしない。

v4 の主判定（evaluate_prospective_v4.py prospective）と同じ 20261113〜20270112・同じ標本の条件で、
v5（v4 の凍結モデル ＋ 展示進入の補正。凍結したのは v5b）を v4 と比べる。v4 の主判定の結論はこの結果で変えない。

  Q1 v5 − v4 の log-loss（1着・2連単・2連複・3連単・3連複）。主: 2連単・3連単
  Q2 v5 単体 − 確定市場（2連単・2連複・3連単・3連複）
  Q3 合成（締切約9分前の板）後の log-loss: v5 − v4、v5 − 確定市場
  Q4 紙上回収率（ev_settle>1、各100円）: v5 / v4、展示の進入 変更あり／枠なり の内訳

実行: .venv/bin/python src/evaluate_prospective_v5.py         （2か月ぶん揃ってから。途中では止まる）
出力: prereg/live_v5_prospective_result.json（上書きしない）
"""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import polars as pl

import evaluate_prospective_v2 as ep2
from holdout import HOLDOUT_END
import live_market_v2 as m2
import live_market_v4 as m4
import live_market_v5 as m5
import live_v2 as v2
import live_v3 as v3
import live_v4 as f4
import v5_entry as v5
import v5b_course as vb

ROOT, KEY, BK = v3.ROOT, v3.KEY, v3.BK
START, END = "20261113", "20270112"
OUT = ROOT / "prereg/live_v5_prospective_result.json"
FREEZE = ROOT / "prereg/live_v5_freeze.json"


def ci(diff, dates):
    return v2.day_bootstrap(diff[:, None], dates)[0]


def block(diff, dates):
    ok = np.isfinite(diff)
    return dict(races=int(ok.sum()), mean=float(diff[ok].mean()), day_bootstrap_95ci=ci(diff[ok], dates[ok]))


def run():
    frozen = json.loads(FREEZE.read_text())["sha256"]
    for path, sha in frozen.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"Frozen artifact changed: {path}")
    if OUT.exists():
        raise SystemExit("Result already recorded; refusing overwrite")
    if START <= HOLDOUT_END:
        raise SystemExit("Evaluation period overlaps the sealed holdout")
    model = f4.load_model(m5.V4_MODEL)
    if START <= model.training_through:
        raise SystemExit("Evaluation period overlaps the v4 training data")
    have = min(max(p.stem for p in (ROOT / "data/raw/boatracecsv/od2").glob("*.csv")),
               max(p.stem for p in (ROOT / "data/raw/boatracecsv/stt").glob("*.csv")),
               pl.read_parquet(ROOT / "data/parsed/results.parquet")["date"].max())
    if have < END:
        raise SystemExit(f"Data reaches {have}; the registered window ends {END}. Not running early")
    entry, coef5 = json.loads(m5.ENTRY.read_text()), json.loads(m5.MARKET.read_text())
    coef4 = json.loads((ROOT / "data/models/live_v4/market_coefficients.json").read_text())

    # 標本: evaluate_prospective_v4.py と同じ（展示タイムが締切前60分以内に出ていて K の値と一致する6艇）
    f3 = v3.load_frame(lo=START, hi=END)
    if f3["date"].max() != END:
        raise SystemExit("v3 feature cache does not reach the evaluation end; run v3_features.py")
    in_window = pl.col("minutes_before_scheduled_deadline").is_between(0, 60, closed="right")
    tk = ep2.raw_previews("tkz", START, END).filter(in_window)
    tenji_pre = pl.concat([tk.select(*KEY, pl.lit(b).cast(pl.Int64).alias("boat"),
                                     pl.col(f"艇{b}_展示タイム").cast(pl.Float64, strict=False).alias("pre"))
                           for b in range(1, 7)])
    f2 = v3.v2_frame(END).filter(pl.col("date") >= START)
    pre_ok = (f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="left").group_by(KEY)
              .agg(ok=pl.col("pre").eq_missing(pl.col("tenji")).all(), n=pl.col("pre").is_not_null().sum())
              .filter(pl.col("ok") & (pl.col("n") == 6)).select(KEY))
    frame = f4.add_ratings(f3.join(pre_ok, on=KEY).sort(BK))
    keys = frame.select(KEY).unique(maintain_order=True)
    dates = keys["date"].to_numpy()
    order = np.argsort(frame["rank"].to_numpy().reshape(-1, 6), axis=1)

    p4 = model.predict(frame)
    course, st = v5.arrays(keys, v5.exhibition(START, END))
    changed = (course != v5.BOATS).any(axis=1)
    p5 = v5.correct(*p4, course, st, entry, vb.extra_for(entry, keys, course))
    l4, l5 = f4.losses(order, *p4), f4.losses(order, *p5)
    market = f4.final_market_losses(keys, order)
    report = dict(period=[START, END], evaluated_at=datetime.now(timezone.utc).isoformat(),
                  races=keys.height, exhibition_changed=int(changed.sum()), entry_variant=entry.get("variant"),
                  v4_training_through=model.training_through)
    report["Q1"] = {pool: dict(all=block(l5[:, j] - l4[:, j], dates),
                               changed=block(np.where(changed, l5[:, j] - l4[:, j], np.nan), dates),
                               v4=float(l4[:, j].mean()), v5=float(l5[:, j].mean()))
                    for j, pool in enumerate(f4.NAMES)}
    report["Q1"]["verdict"] = ("v5 better" if all(report["Q1"][p]["all"]["day_bootstrap_95ci"][1] < 0
                                                  for p in ("2連単", "3連単")) else "not established")
    report["Q2"] = {pool: dict(v5_minus_final_market=block(l5[:, j + 1] - market[:, j], dates),
                               v4_minus_final_market=block(l4[:, j + 1] - market[:, j], dates))
                    for j, pool in enumerate(f4.BET)}

    # Q3・Q4: 板との合成と紙上回収率
    od = ep2.raw_previews("od2", START, END).filter(in_window)
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet").filter(pl.col("date") >= START)
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
        r = np.arange(len(y))
        final_market = -np.log(np.clip((100 / amount) / m2.TAKE[pool], 1e-15, 1))
        loss, ev = {}, {}
        for label, joint, coef in (("v5", p5[1], coef5), ("v4", p4[1], coef4)):
            _, _, P, _, ev[label] = m4.apply(pool, coef, joint[rows], o2t, o2f)
            loss[label] = -np.log(np.clip(P[r, y], 1e-15, 1))
        report["Q3"][pool] = dict(races=len(y), v5_minus_v4=block(loss["v5"] - loss["v4"], day),
                                  v5_minus_final_market=block(loss["v5"] - final_market, day),
                                  v4_minus_final_market=block(loss["v4"] - final_market, day))
        seg = changed[rows]
        report["Q4"][pool] = {label: {name: m2.paper((ev[label] > 1) & mask[:, None], y, amount, day)
                                      for name, mask in (("all", np.ones(len(y), bool)), ("by_lane", ~seg),
                                                         ("changed", seg))}
                              for label in ("v5", "v4")}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("period", "races", "Q1", "Q2")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
