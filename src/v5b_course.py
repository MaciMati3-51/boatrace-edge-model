"""v5b — v5 の展示進入の補正層に、選手×コースの成績差を足す。設計と判定は prereg/live_v5b_design.md。

v5（v5_entry.py）は「コースの価値の平均的なずれ」しか足していなかった。v3/v4 の位置の特徴は全て枠番基準なので、
展示で別コースに入った選手がそのコースで強いかは、どこにも入っていない。v5b はそれを3列足す:

    d_win / d_top2 / d_top3 = logit(選手の 展示コースでの率) − logit(選手の 枠番コースでの率)

率は本番の進入（K の entry_course）の履歴から、対象日より前の出走だけで集計し、全国のコース別の率へ
SHRINK 走ぶん縮小する。展示が枠なりなら 0（v5b も v4 と同じ確率を返す）。展示STは使わない。

係数の推定には、撮影時刻の無い 2026-01〜04 の stt（`backfill-from-daily`、中身は本物の展示値）も使う。
v5 の検証で使った 2026-07-19〜11-12 は2回目なので、ここでの数字は探索扱い。

  evaluate  2026-01-01〜07-18 で推定、07-19〜11-12 で v4・v5 と比べる（1回。結果ファイルがあれば止まる）
  fit       最終係数を 2026-01-01〜11-12 の時系列外予測全部で推定して data/models/live_v5/ に保存
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

import evaluate_v5_entry as ev5
import live_v4 as f4
import v5_entry as v5

ROOT, KEY = v5.ROOT, v5.KEY
RESULT = ROOT / "prereg/live_v5b_check_result.json"
DESIGN = ROOT / "prereg/live_v5b_design.md"
COEF = v5.OUT / "entry_coefficients.json"
EXTRA = ["d_win", "d_top2", "d_top3"]
SHRINK = 20.0
PRIOR_END = "20251231"   # 全国のコース別の率（縮小の先）はこの日までの出走で固定


# ---------------------------------------------------------------- 選手×コースの成績
def course_events(results):
    """1出走1行: 選手・本番の進入コース・1着/2連対/3連対。欠場（K）と進入不明は除く。F・L は着外として数える。"""
    rank = pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8)
    return (results.filter(~pl.col("rank_raw").str.starts_with("K") & pl.col("entry_course").is_not_null())
            .select("date", "racer_id", pl.col("entry_course").alias("course"),
                    win=(rank == 1).fill_null(False).cast(pl.Float64),
                    top2=(rank <= 2).fill_null(False).cast(pl.Float64),
                    top3=(rank <= 3).fill_null(False).cast(pl.Float64)))


def cumulative(ev):
    """(選手, コース, 日) ごとに、その日までの累計（その日を含む）。"""
    daily = ev.group_by("racer_id", "course", "date").agg(n=pl.len(), win=pl.col("win").sum(),
                                                         top2=pl.col("top2").sum(), top3=pl.col("top3").sum())
    return daily.sort("racer_id", "course", "date").with_columns(
        [pl.col(c).cum_sum().over("racer_id", "course") for c in ("n", "win", "top2", "top3")])


def priors(ev):
    base = ev.filter(pl.col("date") <= PRIOR_END).group_by("course").agg(
        [pl.col(c).mean() for c in ("win", "top2", "top3")]).sort("course")
    if base["course"].to_list() != list(range(1, 7)):
        raise ValueError("Course priors need history for courses 1..6 before PRIOR_END")
    return {c: base[c].to_numpy() for c in ("win", "top2", "top3")}       # index = コース − 1


def boats_for(keys, entries=None):
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet",
                              columns=[*KEY, "boat", "racer_id"]) if entries is None else entries
    b = keys.join(entries.select(*KEY, "boat", "racer_id"), on=KEY, how="left").sort([*KEY, "boat"])
    b = keys.with_row_index("order").join(b, on=KEY).sort("order", "boat").drop("order")
    if b.height != 6 * keys.height or b["racer_id"].null_count():
        raise ValueError("Entries missing for some races")
    return b


def exhibition_fit(lo, hi):
    """推定用の stt: 締切前60分以内の撮影 ＋ 撮影時刻の無い後追い取得（backfill-from-daily）。"""
    stt = v5.ep2.raw_previews("stt", lo, hi).filter(
        pl.col("minutes_before_scheduled_deadline").is_between(0, 60, closed="right")
        | (pl.col("取得日時") == "backfill-from-daily"))
    return stt.select(*KEY, *[pl.col(f"艇{b}_コース").cast(pl.Int64, strict=False).alias(f"c{b}") for b in v5.BOATS],
                      *[pl.col(f"艇{b}_スタート展示").cast(pl.Float64, strict=False).alias(f"s{b}") for b in v5.BOATS])


def prepared(paths, lo, hi, ex, results):
    keys, p1, joint, tri, order = v5.load_oos(paths, lo, hi)
    course, st = v5.arrays(keys, ex)
    extra = deltas(boats_for(keys), course, results)
    return keys, p1, joint, tri, order, course, st, extra


def deltas(boats, course, results):
    """boats: KEY + boat + racer_id（keys の順で6艇ずつ、艇番の昇順）。course: (レース, 6) の展示コース。
    返り値 (レース, 6, 3)。各レースとも、その日より前の結果だけを使う（as-of 結合で同じ日は入らない）。"""
    if boats.height != 6 * len(course) or not (boats["boat"].to_numpy().reshape(-1, 6) == v5.BOATS).all():
        raise ValueError("boats must hold boats 1..6 per race in key order")
    ev = course_events(results)
    prior = priors(ev)
    cum = cumulative(ev).sort("date")
    target = boats.with_row_index("row").with_columns(ex=pl.Series(course.ravel()).cast(pl.Int64))
    vals = {}
    for label, col in (("lane", "boat"), ("ex", "ex")):
        t = target.select("row", "date", "racer_id", pl.col(col).alias("course")).sort("date")
        joined = t.join_asof(cum, on="date", by=["racer_id", "course"], strategy="backward",
                             allow_exact_matches=False).sort("row")
        c = joined["course"].to_numpy() - 1
        n = joined["n"].fill_null(0).to_numpy().astype(float)
        vals[label] = {m: (joined[m].fill_null(0).to_numpy() + SHRINK * prior[m][c]) / (n + SHRINK)
                       for m in ("win", "top2", "top3")}
    logit = lambda p: np.log(p) - np.log1p(-p)
    out = np.stack([logit(vals["ex"][m]) - logit(vals["lane"][m]) for m in ("win", "top2", "top3")], axis=-1)
    out = out.reshape(len(course), 6, 3)
    out[course == v5.BOATS] = 0.0          # 枠なりの艇は厳密に 0
    if not np.isfinite(out).all():
        raise ValueError("Non-finite course deltas")
    return out


def extra_for(coef, keys, course, boats=None, results=None):
    """係数が v5b なら選手×コースの差 (レース, 6, 3)、v5 entry なら None。boats 省略時は entries.parquet から。"""
    if not coef.get("extra"):
        return None
    if coef["extra"] != EXTRA or coef.get("shrink", SHRINK) != SHRINK or coef.get("prior_end", PRIOR_END) != PRIOR_END:
        raise ValueError("Coefficients were fit with different course-delta settings")
    boats = boats_for(keys) if boats is None else boats
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet",
                              columns=["date", "racer_id", "entry_course", "rank_raw"]) if results is None else results
    return deltas(boats, course, results)


# ---------------------------------------------------------------- 検証（1回）
def evaluate():
    if RESULT.exists():
        raise SystemExit(f"{RESULT.relative_to(ROOT)} exists; the check runs once")
    design = json.loads(DESIGN.read_text().split("```json")[1].split("```")[0])
    frozen = json.loads((ROOT / "prereg/live_v4_freeze.json").read_text())["sha256"]
    for p in v5.EVAL_OOS:
        if v5.sha(p) != frozen[str(p.relative_to(ROOT))]:
            raise ValueError(f"{p} differs from the frozen v4a predictions")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet",
                              columns=["date", "racer_id", "entry_course", "rank_raw"])
    fk, fp1, fj, ft, fo, fc, fs, fx = prepared([v5.FOLD_OOS], *v5.FIT_WINDOW, exhibition_fit(*v5.FIT_WINDOW), results)
    coef_b = v5.fit(fp1, fj, ft, fo, fc, fs, False, fx, EXTRA)
    coef_a = v5.fit(fp1, fj, ft, fo, fc, fs, False)            # v5 entry を全データで推定し直したもの（比較用）

    keys, p1, joint, tri, order, course, st, extra = prepared(v5.EVAL_OOS, *v5.EVAL_WINDOW,
                                                              v5.exhibition(*v5.EVAL_WINDOW), results)
    dates = keys["date"].to_numpy()
    changed = (course != v5.BOATS).any(axis=1)
    probs = {"v4": (p1, joint, tri), "entry_full": v5.correct(p1, joint, tri, course, st, coef_a),
             "v5b": v5.correct(p1, joint, tri, course, st, coef_b, extra)}
    loss = {k: f4.losses(order, *p) for k, p in probs.items()}
    market = f4.final_market_losses(keys, order)
    pools = f4.NAMES
    report = dict(design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(), fold_oos_sha256=v5.sha(v5.FOLD_OOS),
                  fit_races=int(len(fo)), fit_changed=int((fc != v5.BOATS).any(axis=1).sum()),
                  eval_races=int(len(order)), eval_changed=int(changed.sum()),
                  coefficients=dict(v5b=coef_b, entry_full=coef_a),
                  logloss={k: v.mean(axis=0).tolist() for k, v in loss.items()}, order=pools)
    for label, (a, b) in {"v5b_minus_v4": ("v5b", "v4"), "v5b_minus_entry_full": ("v5b", "entry_full"),
                          "entry_full_minus_v4": ("entry_full", "v4")}.items():
        report[label] = {pool: ev5.paired(loss[a][:, j] - loss[b][:, j], dates) for j, pool in enumerate(pools)}
        report[label + "_changed"] = {pool: ev5.paired(loss[a][changed, j] - loss[b][changed, j], dates[changed])
                                      for j, pool in enumerate(pools)}
    for name in probs:
        report[f"{name}_minus_final_market"] = {
            pool: ev5.paired(loss[name][:, j + 1] - market[:, j], dates) for j, pool in enumerate(f4.BET)}
        report[f"{name}_minus_final_market_changed"] = {
            pool: ev5.paired(np.where(changed, loss[name][:, j + 1] - market[:, j], np.nan), dates)
            for j, pool in enumerate(f4.BET)}
    block = report["v5b_minus_v4"]
    ok = all(block[p]["ci"][1] < 0 and block[p]["first_half"]["mean"] < 0 and block[p]["second_half"]["mean"] < 0
             for p in design["decision_pools"])
    report["decision"] = dict(freeze_v5b=ok)
    ex = v5.exhibition(*v5.EVAL_WINDOW)
    report["paper"] = {name: ev5.paper(keys, *p, ex, name) for name, p in probs.items()}
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    with RESULT.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["decision"]))
    return report


def final_fit():
    """最終係数: 2026-01-01〜11-12 の時系列外予測全部（推定用の stt 条件）。"""
    if not json.loads(RESULT.read_text())["decision"]["freeze_v5b"]:
        raise SystemExit("The check did not support freezing v5b")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet",
                              columns=["date", "racer_id", "entry_course", "rank_raw"])
    _, p1, joint, tri, order, course, st, extra = prepared(
        [v5.FOLD_OOS] + v5.EVAL_OOS, v5.FIT_WINDOW[0], v5.EVAL_WINDOW[1],
        exhibition_fit(v5.FIT_WINDOW[0], v5.EVAL_WINDOW[1]), results)
    coef = v5.fit(p1, joint, tri, order, course, st, False, extra, EXTRA)
    coef.update(variant="v5b", shrink=SHRINK, prior_end=PRIOR_END,
                fitted_on=[v5.FIT_WINDOW[0], v5.EVAL_WINDOW[1]],
                oos_sha256={str(Path(p).relative_to(ROOT)): v5.sha(p) for p in [v5.FOLD_OOS] + v5.EVAL_OOS},
                v4_model_sha256=v5.sha(ROOT / "data/models/live_v4/model.pkl"),
                generated_at=datetime.now(timezone.utc).isoformat())
    v5.OUT.mkdir(parents=True, exist_ok=True)
    COEF.write_text(json.dumps(coef, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(coef, ensure_ascii=False, indent=2))
    return coef


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["evaluate", "fit"])
    args = ap.parse_args()
    evaluate() if args.command == "evaluate" else final_fit()
