"""prereg/live_v5_design.md §3 の検証（1回だけ）。v5_entry.py evaluate から呼ぶ。

係数は 2026-01-01〜07-18（live_v4_y25 = 学習 〜20251231 の時系列外予測）で推定し、2026-07-19〜11-12
（live_v4a の凍結済み時系列外予測）で v4 と比べる。紙上回収率は市場合成・変動補正を 07-19〜09-09 で
v4・v5 とも推定し直して 09-10〜11-12 で数える（exacta_marginals.py と同じ分け方）。
出力: prereg/live_v5_check_result.json（あれば止まる）
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

import live_market_v4 as m4
from live_market import roi_ci
import live_v2 as v2
import live_v4 as f4
import v5_entry as v5

ROOT, KEY = v5.ROOT, v5.KEY
RESULT = ROOT / "prereg/live_v5_check_result.json"
DESIGN = ROOT / "prereg/live_v5_design.md"
HALF = "20260915"                     # 前半 07-19〜09-14 / 後半 09-15〜11-12
MARKET_FIT = ("20260719", "20260909")
MARKET_TEST = ("20260910", "20261112")
m2, m3 = m4.m2, m4.m3


def paired(diff, dates):
    ok = np.isfinite(diff)
    d, t = diff[ok], dates[ok]
    halves = {}
    for name, mask in (("first_half", t < HALF), ("second_half", t >= HALF)):
        halves[name] = dict(n=int(mask.sum()), mean=float(d[mask].mean()))
    return dict(n=int(ok.sum()), mean=float(d.mean()), ci=v2.day_bootstrap(d[:, None], t)[0], **halves)


def paper(keys, p1, joint, tri, ex, label):
    """市場合成・変動補正を MARKET_FIT で推定し、MARKET_TEST で ev_settle>1 の各100円を数える。"""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{label}.parquet"
        rank = pl.read_parquet(v5.EVAL_OOS[0], columns=KEY + ["rank"])
        rank = pl.concat([rank, pl.read_parquet(v5.EVAL_OOS[1], columns=KEY + ["rank"])])
        keys.with_columns(joint=pl.Series(joint.tolist())).join(rank, on=KEY).write_parquet(path)
        data = m3.load_dataset(path, m4.boards(MARKET_FIT[0], MARKET_TEST[1]))
    for pool, d in data.items():
        dates = d["dates"]
        fit_mask = (dates >= MARKET_FIT[0]) & (dates <= MARKET_FIT[1])
        test = (dates >= MARKET_TEST[0]) & (dates <= MARKET_TEST[1])
        coef = {pool: m3.fit_pool(pool, d, fit_mask)}
        p = d["p"][test]
        o2t, o2f = d["o2t"][test], d["o2f"][test]
        market = m2.market_inputs(pool, o2t, o2f)
        final = m2.distribution([np.log(np.clip(p, 1e-15, 1))] + market, coef[pool]["combined"])
        board = o2t if pool == "2t" else o2f
        ev = final * board * m2.settle_factor(m2.drift_design(pool, final, o2t, o2f), coef[pool]["drift"])
        y, amount, td = d["y"][test], d["amount"][test], dates[test]
        r = np.arange(len(y))
        course, _ = v5.arrays(d["keys"].filter(pl.Series(test)), ex)
        changed = (course != v5.BOATS).any(axis=1)
        res = dict(combined=coef[pool]["combined"], races=int(test.sum()),
                   log_loss=float(-np.log(final[r, y]).mean()))
        for name, mask in (("all", np.ones(len(y), bool)), ("by_lane", ~changed), ("changed", changed)):
            bets = (ev > 1) & mask[:, None]
            stakes = bets.sum(axis=1) * 100.0
            returns = bets[r, y] * amount
            res[name] = dict(races=int(mask.sum()), bets=int(bets.sum()),
                             roi=float(returns.sum() / max(stakes.sum(), 1)),
                             roi_ci=roi_ci(stakes, returns, td) if stakes.sum() else None)
        out[pool] = res
    return out


def run(rehearsal=None):
    """rehearsal: 配線の試運転。係数をゼロ（= v4 と同じ確率）に置き、結果は rehearsal のパスへ。判定に使わない。"""
    result = Path(rehearsal) if rehearsal else RESULT
    if result.exists():
        raise SystemExit(f"{result} exists; the registered check runs once")
    text = DESIGN.read_text()
    design = json.loads(text.split("```json")[1].split("```")[0])
    frozen = json.loads((ROOT / "prereg/live_v4_freeze.json").read_text())["sha256"]
    for p in v5.EVAL_OOS:
        if v5.sha(p) != frozen[str(p.relative_to(ROOT))]:
            raise ValueError(f"{p} differs from the frozen v4a predictions")

    # 係数の推定（2026-01-01〜07-18。封印期間は学習にだけ使い、ここで評価値は出さない）
    if rehearsal:
        fo, fc = np.zeros((0, 6), int), np.zeros((0, 6), int)
        coefs = {name: dict(use_st=use_st, beta=[[0.0] * len(v5.names(use_st, k)) for k in (1, 2, 3)])
                 for name, use_st in (("entry", False), ("entry_st", True))}
    else:
        fk, fp1, fj, ft, fo = v5.load_oos([v5.FOLD_OOS], *v5.FIT_WINDOW)
        fc, fs = v5.arrays(fk, v5.exhibition(*v5.FIT_WINDOW))
        coefs = {name: v5.fit(fp1, fj, ft, fo, fc, fs, use_st)
                 for name, use_st in (("entry", False), ("entry_st", True))}

    # 検証（2026-07-19〜11-12）
    keys, p1, joint, tri, order = v5.load_oos(v5.EVAL_OOS, *v5.EVAL_WINDOW)
    ex = v5.exhibition(*v5.EVAL_WINDOW)
    course, st = v5.arrays(keys, ex)
    dates = keys["date"].to_numpy()
    changed = (course != v5.BOATS).any(axis=1)
    probs = {"v4": (p1, joint, tri)}
    for name, coef in coefs.items():
        probs[name] = v5.correct(p1, joint, tri, course, st, coef)
    loss = {name: f4.losses(order, *p) for name, p in probs.items()}
    market = f4.final_market_losses(keys, order)
    pools = f4.NAMES                                   # 1着, 2連単, 2連複, 3連単, 3連複
    report = dict(
        design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        fold_oos_sha256=None if rehearsal else v5.sha(v5.FOLD_OOS), fit_window=list(v5.FIT_WINDOW), eval_window=list(v5.EVAL_WINDOW),
        fit_races=int(len(fo)), fit_changed=int((fc != v5.BOATS).any(axis=1).sum()),
        eval_races=int(len(order)), eval_changed=int(changed.sum()),
        exhibition_st_missing=int(np.isnan(st).any(axis=1).sum()),
        coefficients=coefs, logloss={k: v.mean(axis=0).tolist() for k, v in loss.items()}, order=pools)
    comparisons = {"entry_minus_v4": ("entry", "v4"), "entry_st_minus_entry": ("entry_st", "entry"),
                   "entry_st_minus_v4": ("entry_st", "v4")}
    for label, (a, b) in comparisons.items():
        report[label] = {pool: paired(loss[a][:, j] - loss[b][:, j], dates) for j, pool in enumerate(pools)}
        report[label + "_changed"] = {pool: paired(loss[a][changed, j] - loss[b][changed, j], dates[changed])
                                      for j, pool in enumerate(pools)}
        report[label + "_by_lane"] = {pool: paired(loss[a][~changed, j] - loss[b][~changed, j], dates[~changed])
                                      for j, pool in enumerate(pools)}
    for name in probs:
        report[f"{name}_minus_final_market"] = {
            pool: paired(loss[name][:, j + 1] - market[:, j], dates) for j, pool in enumerate(f4.BET)}
        report[f"{name}_minus_final_market_changed"] = {
            pool: paired(np.where(changed, loss[name][:, j + 1] - market[:, j], np.nan), dates)
            for j, pool in enumerate(f4.BET)}

    # 事前登録の判定
    def passes(block):
        return all(block[pool]["ci"][1] < 0 and block[pool]["first_half"]["mean"] < 0
                   and block[pool]["second_half"]["mean"] < 0 for pool in design["decision_pools"])
    entry_ok = passes(report["entry_minus_v4"])
    st_ok = entry_ok and passes(report["entry_st_minus_entry"])
    report["decision"] = dict(entry_adopted=entry_ok, st_adopted=st_ok,
                              v5="entry_st" if st_ok else ("entry" if entry_ok else None))

    # 副指標: 紙上回収率（判定には使わない）
    report["paper"] = {name: paper(keys, *p, ex, name) for name, p in probs.items()}
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    with result.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if rehearsal:
        same = all(np.allclose(loss["v4"], loss[k], rtol=0, atol=1e-12) for k in coefs)
        print("rehearsal: zero coefficients reproduce v4 exactly:", same,
              {k: {pool: v["all"]["bets"] for pool, v in report["paper"][k].items()} for k in report["paper"]})
        return report
    print(json.dumps({k: report[k] for k in ("decision", "logloss", "entry_minus_v4", "entry_st_minus_entry",
                                             "entry_minus_final_market")}, ensure_ascii=False, indent=2))
    return report
