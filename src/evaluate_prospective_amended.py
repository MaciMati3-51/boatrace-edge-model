"""prereg/prospective_amendment_20261122.md の前向き検証（v4 の主判定・v5b の副判定）。1回だけ。

BoatraceCSV が 2026-11-15 に公開を停止し、20261113 以降の od2（締切前の板）・stt・tkz が無くなった。
事前登録の evaluate_prospective_v4.py・evaluate_prospective_v5.py はそれらを前提にしていて回せないので、
次の2点だけ差し替える。モデル・係数・判定の式は変えない（凍結ハッシュを照合する）。

  1. 標本: 「締切前60分以内の展示タイムが K と6艇とも一致」→「boatraceopenapi の直前情報の展示タイムが K と6艇とも一致」
  2. 展示の進入・ST（v5b の入力）: stt → boatraceopenapi の直前情報（2026-10-01〜10 の 1,536R で stt と全艇一致を確認）

板との合成（Q3）と紙上回収率（Q4）は締切前の板が無いので測れない（「測れない」と記録する）。副指標として、turnmark の
確定オッズ全組から正規化子を仮置きしない確定市場の log-loss を足す。

実行: .venv/bin/python src/evaluate_prospective_amended.py          （20270112 まで揃ってから。途中では止まる）
出力: prereg/prospective_amended_result.json（上書きしない）
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

from holdout import HOLDOUT_END
import live_model as v1
import live_v2 as v2
import live_v3 as v3
import live_v4 as f4
import parse_openapi as po
import v5_entry as v5
import v5b_course as vb

ROOT, KEY, BK = v3.ROOT, v3.KEY, v3.BK
START, END = "20261113", "20270112"
OUT = ROOT / "prereg/prospective_amended_result.json"
AMENDMENT = ROOT / "prereg/prospective_amendment_20261122.md"
FREEZES = [ROOT / "prereg/live_v4_freeze.json", ROOT / "prereg/live_v5_freeze.json",
           ROOT / "prereg/prospective_amendment_20261122_freeze.json"]
MODELS = dict(v3=v3.OUT / "model.pkl", v4=ROOT / "data/models/live_v4/model.pkl")


def block(diff, dates):
    ok = np.isfinite(diff)
    return dict(races=int(ok.sum()), mean=float(diff[ok].mean()),
                day_bootstrap_95ci=v2.day_bootstrap(diff[ok][:, None], dates[ok])[0])


def exact_market_losses(keys, order, lo, hi):
    """turnmark の確定オッズ全組から q = (1/o) / Σ(1/o)。欠場で空欄の組があるレース・的中目のオッズが 0 か無いレースは nan。列は f4.BET の順。"""
    odds = po.final_odds(lo, hi)
    # オッズ 0 は票の入らなかった組（市場の確率 0）。Σ(1/オッズ) に入れない。的中目が 0 のレースは nan（特払）
    agg = odds.group_by([*KEY, "pool"]).agg(inv=(1 / pl.col("odds").filter(pl.col("odds") > 0)).sum(),
                                            nulls=pl.col("odds").is_null().sum(), n=pl.len())
    out = np.full((keys.height, len(f4.BET)), np.nan)
    o = order[:, :3] + 1
    for j, (_, pool) in enumerate(f4.BET.values()):
        if pool == "2t":
            win = [f"{a}-{b}" for a, b, _ in o]
        elif pool == "2f":
            win = [f"{min(a, b)}-{max(a, b)}" for a, b, _ in o]
        elif pool == "3t":
            win = [f"{a}-{b}-{c}" for a, b, c in o]
        else:
            win = ["-".join(map(str, sorted(t))) for t in o]
        size = {"2t": 30, "2f": 15, "3t": 120, "3f": 20}[pool]
        w = keys.with_columns(pool=pl.lit(pool), combo=pl.Series(win))
        w = (w.join(odds, on=[*KEY, "pool", "combo"], how="left", maintain_order="left")
             .join(agg, on=[*KEY, "pool"], how="left", maintain_order="left"))
        ok = ((w["nulls"].fill_null(1).to_numpy() == 0) & (w["n"].fill_null(0).to_numpy() == size)
              & (w["odds"].fill_null(0.0).to_numpy() > 0))
        q = (1 / w["odds"].fill_null(np.nan).to_numpy()) / w["inv"].fill_null(np.nan).to_numpy()
        out[ok, j] = -np.log(q[ok])
    return out


def run(rehearsal=None):
    """rehearsal=(lo, hi, out, v4 の model.pkl, v3 の model.pkl): 配線の試運転。凍結・期間の照合を飛ばす。判定に使わない。"""
    lo, hi, out = (START, END, OUT) if rehearsal is None else (rehearsal[0], rehearsal[1], Path(rehearsal[2]))
    models = MODELS if rehearsal is None else dict(v4=Path(rehearsal[3]), v3=Path(rehearsal[4]))
    if out.exists():
        raise SystemExit("Result already recorded; refusing overwrite")
    if rehearsal is None:
        for freeze in FREEZES:
            for path, sha in json.loads(freeze.read_text())["sha256"].items():
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
                    raise SystemExit(f"Frozen artifact changed: {path} ({freeze.name})")
        if START <= HOLDOUT_END:
            raise SystemExit("Evaluation period overlaps the sealed holdout")
        have = min(pl.read_parquet(ROOT / "data/parsed/results.parquet")["date"].max(),
                   max(p.stem for p in (ROOT / "data/raw/openapi").glob("2*.json")),
                   max(p.stem for p in (ROOT / "data/raw/turnmark").glob("2*.json")))
        if have < END:
            raise SystemExit(f"Data reaches {have}; the registered window ends {END}. Not running early")
    m4 = f4.load_model(models["v4"])
    if rehearsal is None and lo <= m4.training_through:
        raise SystemExit("Evaluation period overlaps the v4 training data")
    entry = json.loads(vb.COEF.read_text())

    f3 = v3.load_frame(lo=lo, hi=hi)
    if f3["date"].max() != hi:
        raise SystemExit("v3 feature cache does not reach the evaluation end; run v3_features.py")
    preview = po.previews(lo, hi)
    tenji_pre = pl.concat([preview.select(*KEY, pl.lit(b).cast(pl.Int64).alias("boat"), pl.col(f"t{b}").alias("pre"))
                           for b in po.BOATS])
    f2 = v3.v2_frame(hi).filter(pl.col("date") >= lo)       # 標本を v3 の検証と揃えるためだけに使う
    pre_ok = (f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="left").group_by(KEY)
              .agg(ok=pl.col("pre").eq_missing(pl.col("tenji")).all(), n=pl.col("pre").is_not_null().sum())
              .filter(pl.col("ok") & (pl.col("n") == 6)).select(KEY))
    mismatch = f2.select(*BK, "tenji").join(tenji_pre, on=BK).filter(pl.col("pre") != pl.col("tenji")).height
    f3 = f3.join(pre_ok, on=KEY).sort(BK)
    frame = f4.add_ratings(f3)
    keys = f3.select(KEY).unique(maintain_order=True)
    if not frame.select(KEY).unique(maintain_order=True).equals(keys):
        raise SystemExit("v3 and v4 race sets differ")
    dates = keys["date"].to_numpy()
    order = np.argsort(f3["rank"].to_numpy().reshape(-1, 6), axis=1)

    p1_3, j3 = v3.load_model(models["v3"]).predict(f3)
    p4 = m4.predict(frame)
    course, st = v5.arrays(keys, preview)
    changed = (course != v5.BOATS).any(axis=1)
    p5 = v5.correct(*p4, course, st, entry, vb.extra_for(entry, keys, course))
    l3, l4, l5 = v1.losses(f3, p1_3, j3), f4.losses(order, *p4), f4.losses(order, *p5)
    market = f4.final_market_losses(keys, order)
    exact = exact_market_losses(keys, order, lo, hi)

    report = dict(amendment_sha256=hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() if AMENDMENT.exists() else None,
                  period=[lo, hi], evaluated_at=datetime.now(timezone.utc).isoformat(), rehearsal=bool(rehearsal),
                  races=keys.height, preview_tenji_vs_k_mismatched_boats=mismatch,
                  exhibition_changed=int(changed.sum()), v4_training_through=m4.training_through)
    # v4（主判定、prereg/live_v4_design.md §4 の Q1・Q2）
    report["v4_Q1"] = {pool: dict(v4_minus_v3=block(l4[:, j] - l3[:, j], dates), v3=float(l3[:, j].mean()),
                                  v4=float(l4[:, j].mean())) for j, pool in enumerate(f4.NAMES[:3])}
    report["v4_Q1"]["verdict"] = ("v4 better" if all(report["v4_Q1"][p]["v4_minus_v3"]["day_bootstrap_95ci"][1] < 0
                                                     for p in ("2連単", "2連複")) else "not established")
    report["v4_Q2"] = {pool: dict(v4_minus_final_market=block(l4[:, j + 1] - market[:, j], dates),
                                  v4_minus_exact_final_market=block(l4[:, j + 1] - exact[:, j], dates),
                                  v3_minus_final_market=block(l3[:, j + 1] - market[:, j], dates) if j < 2 else None)
                       for j, pool in enumerate(f4.BET)}
    report["v4_Q2"]["verdict"] = {pool: ("v4 alone beats the final market"
                                         if q["v4_minus_final_market"]["day_bootstrap_95ci"][1] < 0 else "not established")
                                  for pool, q in report["v4_Q2"].items()}
    # v5b（副判定、prereg/live_v5b_design.md §4）
    report["v5b_Q1"] = {pool: dict(all=block(l5[:, j] - l4[:, j], dates),
                                   changed=block(np.where(changed, l5[:, j] - l4[:, j], np.nan), dates))
                        for j, pool in enumerate(f4.NAMES)}
    report["v5b_Q1"]["verdict"] = ("v5b better" if all(report["v5b_Q1"][p]["all"]["day_bootstrap_95ci"][1] < 0
                                                       for p in ("2連単", "3連単")) else "not established")
    report["v5b_Q2"] = {pool: dict(v5b_minus_final_market=block(l5[:, j + 1] - market[:, j], dates),
                                   v5b_minus_exact_final_market=block(l5[:, j + 1] - exact[:, j], dates))
                        for j, pool in enumerate(f4.BET)}
    report["not_evaluable"] = ("Q3（締切前の板との合成）と Q4（紙上回収率）: BoatraceCSV の公開停止で 20261113 以降の od2 が無い")
    with out.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if rehearsal is None:
        print(json.dumps({k: report[k] for k in ("period", "races", "v4_Q1", "v5b_Q1")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("usage: evaluate_prospective_amended.py")
    run()
