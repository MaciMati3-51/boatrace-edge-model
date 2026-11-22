"""prereg/live_v4_design.md §4 の検証。v3・v4 とも再学習・係数変更はしない。登録済みの2本だけを回せる。

  check        v4a（学習 〜20260909。v3 と同じ打ち切り）を 20260910〜20261112 で v3 と比べる。
               この区間は v3 の前向き検証で既に開けてある（v4 の設計には使っていない）。1回だけ
  prospective  v4（学習 〜20261112）を 20261113〜20270112 の、凍結後に初めて取得するデータで v3 と比べる。主判定。
               2か月ぶんが揃ってから1回だけ（途中経過を見るための実行はしない）

どちらも: 凍結ハッシュの照合 → v3 と v4 を同じレースで予測 → 確定払戻・締切約9分前の板（od2）と照合。
標本の条件は v3 の前向き検証（evaluate_prospective_v3.py）と同じ。

実行: .venv/bin/python src/evaluate_prospective_v4.py check
      .venv/bin/python src/evaluate_prospective_v4.py prospective
出力: prereg/live_v4_check_result.json / prereg/live_v4_prospective_result.json（上書きしない）
"""
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import polars as pl

import evaluate_prospective_v2 as ep2
from holdout import HOLDOUT_END
import live_market_v2 as m2
import live_market_v4 as m4
import live_model as v1
import live_v2 as v2
import live_v3 as v3
import live_v4 as v4

ROOT, KEY, BK = v3.ROOT, v3.KEY, v3.BK
RUNS = {
    "check": dict(tag="live_v4a", start="20260910", fresh="20260923", end="20261112",
                  out="prereg/live_v4_check_result.json", freeze="prereg/live_v4a_freeze.json"),
    "prospective": dict(tag="live_v4", start="20261113", fresh="20261113", end="20270112",
                        out="prereg/live_v4_prospective_result.json", freeze="prereg/live_v4_freeze.json"),
}
THRESHOLDS = (1.0, 1.1, 1.2)
LOW_ODDS = 10.0  # 副指標: 撮影オッズがこの倍率未満の目だけ（裾に頼らない部分）


def ci(diff, dates):
    return v2.day_bootstrap(diff if diff.ndim == 2 else diff[:, None], dates)


def run(name, rehearsal=None):
    """rehearsal=(tag, start, fresh, end, out) は配線の試運転（凍結照合を飛ばす。数値を判定に使わない）。"""
    cfg = dict(RUNS[name])
    if rehearsal:
        cfg.update(rehearsal)
    tag, start, fresh = cfg["tag"], cfg["start"], cfg["fresh"]
    out, model_dir = ROOT / cfg["out"], ROOT / "data/models" / cfg["tag"]

    def segments(dates):
        seg = {"all": np.ones(len(dates), bool)}
        if fresh > start:
            seg.update(before_fresh=dates < fresh, fresh=dates >= fresh)
        return seg

    main = "fresh" if fresh > start else "all"
    if not rehearsal:
        frozen = json.loads((ROOT / cfg["freeze"]).read_text())["sha256"]
        for path, sha in frozen.items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
                raise SystemExit(f"Frozen artifact changed: {path}")
    if out.exists():
        raise SystemExit("Result already recorded; refusing overwrite")
    if start <= HOLDOUT_END:
        raise SystemExit("Evaluation period overlaps the sealed holdout")
    model = v4.load_model(model_dir / "model.pkl")
    if start <= model.training_through and not rehearsal:
        raise SystemExit("Evaluation period overlaps the v4 training data")
    end = cfg["end"]
    have = min(max(p.stem for p in (ROOT / "data/raw/boatracecsv/od2").glob("*.csv")),
               pl.read_parquet(ROOT / "data/parsed/results.parquet")["date"].max())
    if have < end:
        raise SystemExit(f"Data reaches {have}; the registered window ends {end}. Not running early")
    f3 = v3.load_frame(lo=start, hi=end)
    if f3["date"].max() != end:
        raise SystemExit("v3 feature cache does not reach the evaluation end; run v3_features.py")
    report = dict(run=name, model=tag, period=[start, end], fresh_from=fresh, main_segment=main,
                  evaluated_at=datetime.now(timezone.utc).isoformat(), frozen_hashes_verified=not rehearsal,
                  rehearsal=bool(rehearsal), v4_training_through=model.training_through)

    # 展示タイムが締切前60分以内に出ていて、K文書の値と一致するレースだけ（v3 の前向き検証と同じ条件・同じ手順）
    frames = {k: ep2.raw_previews(k, start, end) for k in ("tkz", "od2")}
    in_window = ((pl.col("minutes_before_scheduled_deadline") > 0)
                 & (pl.col("minutes_before_scheduled_deadline") <= 60))
    tk = frames["tkz"].filter(in_window)
    tenji_pre = pl.concat([tk.select(*KEY, pl.lit(b).cast(pl.Int64).alias("boat"),
                                     pl.col(f"艇{b}_展示タイム").cast(pl.Float64, strict=False).alias("pre"))
                           for b in range(1, 7)])
    f2 = v3.v2_frame(end).filter(pl.col("date") >= start)  # 標本を v3 の検証と揃えるためだけに使う
    pre_ok = (f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="left").group_by(KEY)
              .agg(ok=pl.col("pre").eq_missing(pl.col("tenji")).all(), n=pl.col("pre").is_not_null().sum())
              .filter(pl.col("ok") & (pl.col("n") == 6)).select(KEY))
    mismatch = f2.select(*BK, "tenji").join(tenji_pre, on=BK).filter(pl.col("pre") != pl.col("tenji")).height
    f3 = f3.join(pre_ok, on=KEY).sort(BK)
    f4 = v4.add_ratings(f3)
    keys = f3.select(KEY).unique(maintain_order=True)
    if not f4.select(KEY).unique(maintain_order=True).equals(keys):
        raise SystemExit("v3 and v4 race sets differ")
    report["races"] = dict(predictable=keys.height, tenji_preview_vs_k_mismatched_boats=mismatch)

    p1_3, j3 = v3.load_model(v3.OUT / "model.pkl").predict(f3)
    p1_4, j4, tri = model.predict(f4)
    order = np.argsort(f3["rank"].to_numpy().reshape(-1, 6), axis=1)
    l3, l4 = v1.losses(f3, p1_3, j3), v4.losses(order, p1_4, j4, tri)
    dates = keys["date"].to_numpy()
    market = v4.final_market_losses(keys, order)

    # Q1: v4 − v3 / Q2: 単体 − 確定市場（2連単・2連複・3連単・3連複）
    report["Q1"], report["Q2"] = {}, {}
    for seg_name, seg in segments(dates).items():
        if not seg.any():
            continue
        d = (l4[:, :3] - l3)[seg]
        report["Q1"][seg_name] = dict(races=int(seg.sum()), order=v4.NAMES[:3], v3=l3[seg].mean(axis=0).tolist(),
                                      v4=l4[seg, :3].mean(axis=0).tolist(), v4_minus_v3=d.mean(axis=0).tolist(),
                                      day_bootstrap_95ci=ci(d, dates[seg]))
        q2 = {}
        for j, pool in enumerate(v4.BET):
            ok = seg & np.isfinite(market[:, j])
            q2[pool] = dict(races=int(ok.sum()), final_market=float(market[ok, j].mean()),
                            v4=float(l4[ok, j + 1].mean()))
            diff = l4[ok, j + 1] - market[ok, j]
            q2[pool]["v4_minus_final_market"] = dict(mean=float(diff.mean()), day_bootstrap_95ci=ci(diff, dates[ok])[0])
            if j < 2:
                diff = l3[ok, j + 1] - market[ok, j]
                q2[pool]["v3_minus_final_market"] = dict(mean=float(diff.mean()),
                                                         day_bootstrap_95ci=ci(diff, dates[ok])[0])
        report["Q2"][seg_name] = q2
    q1 = report["Q1"][main]["day_bootstrap_95ci"]
    report["Q1"]["verdict"] = "v4 better" if q1[1][1] < 0 and q1[2][1] < 0 else "not established"
    report["Q2"]["verdict"] = {pool: ("v4 alone beats the final market"
                                      if q["v4_minus_final_market"]["day_bootstrap_95ci"][1] < 0
                                      else "not established") for pool, q in report["Q2"][main].items()}
    report["Q2"]["note"] = ("3連複は完全板が無く、控除の正規化子 Σ(1/オッズ) を 2連複の実測 1.345 で仮置きしている。"
                            "1.333（理論下限）なら市場の log-loss は 0.009 小さい。判定はこの不確かさ込みで読む")

    # Q3・Q4: 締切約9分前の板との合成、確定オッズ期待値での紙上回収率
    od = frames["od2"].filter(in_window)
    report["minutes_before_deadline_od2"] = np.quantile(
        od["minutes_before_scheduled_deadline"].to_numpy(), [.05, .5, .95]).tolist()
    coefs = {"v4": json.loads((model_dir / "market_coefficients.json").read_text()),
             "v3": json.loads((v3.OUT / "market_coefficients.json").read_text())}
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
        for label, joint in (("v4", j4), ("v3", j3)):
            _, _, P, _, ev[label] = m4.apply(pool, coefs[label], joint[rows], o2t, o2f)
            loss[label] = -np.log(np.clip(P[r, y], 1e-15, 1))
        captured = -np.log(m2.normalize_inverse(board)[r, y])
        q3, q4 = {}, {}
        for seg_name, seg in segments(day).items():
            if not seg.any():
                continue
            item = dict(races=int(seg.sum()), final_market=float(final_market[seg].mean()),
                        captured_market=float(captured[seg].mean()))
            for label in ("v4", "v3"):
                diff = (loss[label] - final_market)[seg]
                item[f"{label}_combined"] = float(loss[label][seg].mean())
                item[f"{label}_combined_minus_final_market"] = dict(mean=float(diff.mean()),
                                                                    day_bootstrap_95ci=ci(diff, day[seg])[0])
            diff = (loss["v4"] - loss["v3"])[seg]
            item["v4_combined_minus_v3_combined"] = dict(mean=float(diff.mean()),
                                                         day_bootstrap_95ci=ci(diff, day[seg])[0])
            q3[seg_name] = item
            q4[seg_name] = {}
            for label in ("v4", "v3"):
                rules = {f"ev_settle_gt_{t}": m2.paper(ev[label][seg] > t, y[seg], amount[seg], day[seg])
                         for t in THRESHOLDS}
                low = (ev[label][seg] > 1.0) & (board[seg] < LOW_ODDS)
                rules[f"ev_settle_gt_1.0_odds_lt_{LOW_ODDS:g}"] = m2.paper(low, y[seg], amount[seg], day[seg])
                chosen = ev[label][seg] > 1.0
                rules["model_expected_roi_gt_1.0"] = float(ev[label][seg][chosen].mean()) if chosen.any() else None
                q4[seg_name][label] = rules
        report["Q3"][pool], report["Q4"][pool] = q3, q4
        print(pool, json.dumps(q3, ensure_ascii=False), flush=True)
    report["Q4"]["verdict"] = {}
    for pool in m2.POOLS:
        lo = report["Q4"][pool][main]["v4"]["ev_settle_gt_1.0"]["day_bootstrap_roi_95ci"]
        report["Q4"]["verdict"][pool] = "paper ROI above 100%" if lo and lo[0] > 1 else "not established"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("period", "races", "Q1", "Q2")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in RUNS:
        raise SystemExit("usage: evaluate_prospective_v4.py check|prospective")
    run(sys.argv[1])
