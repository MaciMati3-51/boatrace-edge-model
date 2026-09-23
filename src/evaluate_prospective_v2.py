"""prereg/live_v2_design.md §4 の前向き検証（Q1〜Q4）。再学習・係数変更はしない。

1. 凍結ハッシュを照合し、対象日が封印期間より後であることを確認する
2. boatracecsv の生CSVを audit_previews_2026.py / live_model.load_previews() と同じ変換で読む。
   先に 20260719〜0909 で v1 の dataset.parquet の直前情報列と完全一致することを確かめる
3. v1（凍結）と v2（凍結）で予測し、確定払戻と照合する

実行: OMP_NUM_THREADS=6 .venv/bin/python src/evaluate_prospective_v2.py
出力: prereg/live_v2_prospective_result.json（上書きしない）
"""
from datetime import datetime, timedelta, timezone
from itertools import groupby
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from build_features import racer_history, venue_boat_history
from holdout import HOLDOUT_END
import live_market as m1
import live_market_v2 as m2
import live_model as v1
import live_v2 as v2

ROOT = v2.ROOT
KEY, BK = v2.KEY, v2.BK
START = "20260910"
OUT = ROOT / "prereg/live_v2_prospective_result.json"
FROZEN = {
    "data/models/live_v2/model.pkl": "9c986b92445b5e741451736e892d1f39862f882ac47f3b99f5e4c3ae30e7bfe9",
    "data/models/live_v2/market_coefficients.json": "3031d5cc01d840c24c91ab977aa68eda17f3afdfa92a51fad48607e609a7778f",
    "src/live_v2.py": "95f6dd5d049833655022ce92ede486729c50f29e63d63e78fb52ddefce1bb656",
    "src/live_market_v2.py": "d30cb0f77e395ba7d2f006cc4cc4e2f55ffbebea0245dd131f03f1ba893e6dab",
    "data/models/live_v1/preview_model.pkl": "17442e4f5ce42b53e16dbbb8ce96c18d20b63ecd2500d69533644388a1b1d2e8",
    "data/models/live_v1/market_coefficients.json": "aaab8eb75522f9407de5573525e75e325aaf3e5e99a9db5f0e0967452361eae8",
}


def raw_previews(kind, lo, hi):
    """audit_previews_2026.py と同じ: レースコード→キー、予定締切までの分数。"""
    frames = []
    for path in sorted((ROOT / "data/raw/boatracecsv" / kind).glob("*.csv")):
        if lo <= path.stem <= hi:
            frames.append(pl.read_csv(path, infer_schema=False))
    df = pl.concat(frames, how="diagonal").with_columns(
        date=pl.col("レースコード").str.slice(0, 8),
        jcd=pl.col("レースコード").str.slice(8, 2).cast(pl.Int64),
        race_no=pl.col("レースコード").str.slice(10, 2).cast(pl.Int64))
    if df.select(KEY).unique().height != df.height:
        raise ValueError(f"Duplicate keys {kind}")
    minutes = []
    for race_date, deadline, captured in df.select("date", "締切時刻", "取得日時").iter_rows():
        try:
            limit = datetime.strptime(race_date + " " + deadline, "%Y%m%d %H:%M").replace(
                tzinfo=timezone(timedelta(hours=9)))
            stamp = datetime.fromisoformat(captured)
            if stamp.tzinfo is None:
                raise ValueError("timezone missing")
            minutes.append((limit - stamp).total_seconds() / 60)
        except (TypeError, ValueError):
            minutes.append(None)
    return df.with_columns(pl.Series("minutes_before_scheduled_deadline", minutes, dtype=pl.Float64))


def v1_previews(frames):
    """live_model.load_previews() と同じ列変換（期間の絞り込みは呼び出し側）。"""
    frames = {k: f.filter((pl.col("minutes_before_scheduled_deadline") > 0)
                          & (pl.col("minutes_before_scheduled_deadline") <= 60)) for k, f in frames.items()}
    tk = {"exhibition_weight": "体重(kg)", "adjustment_weight": "体重調整(kg)", "tenji": "展示タイム", "tilt": "チルト"}
    st = {"exhibition_course": "コース", "exhibition_st": "スタート展示"}
    long = []
    for kind, mapping in (("tkz", tk), ("stt", st)):
        boats = []
        for b in range(1, 7):
            boats.append(frames[kind].select(
                *KEY, pl.lit(b).cast(pl.Int64).alias("boat"), pl.col("取得日時").alias(f"{kind}_captured_at"),
                *[pl.col(f"艇{b}_{source}").cast(pl.Float64, strict=False).alias(name) for name, source in mapping.items()],
                *([pl.concat_str([pl.col("date").str.strptime(pl.Date, "%Y%m%d").cast(pl.String),
                                  pl.lit("T"), pl.col("締切時刻"), pl.lit(":00+09:00")]).alias("deadline_at")]
                  if kind == "tkz" else [])))
        long.append(pl.concat(boats))
    weather = {"wind_speed": "風速(m)", "wind_direction": "風向", "wave_height": "波の高さ(cm)",
               "weather_code": "天候", "air_temperature": "気温(℃)", "water_temperature": "水温(℃)"}
    w = frames["sui"].select(*KEY, pl.col("取得日時").alias("sui_captured_at"),
                             *[pl.col(v).cast(pl.Float64, strict=False).alias(k) for k, v in weather.items()])
    return long[0].join(long[1], on=BK, validate="1:1").join(w, on=KEY, validate="m:1")


def check_v1_reproduction():
    frames = {k: raw_previews(k, v1.TRAIN_START, v1.END) for k in ("tkz", "stt", "sui")}
    mine = v1_previews(frames)
    saved = pl.read_parquet(v1.OUT / "dataset.parquet")
    cols = v1.RAW + v1.TIMES + ["deadline_at"]
    joined = saved.select(BK + cols).join(mine.select(BK + cols), on=BK, how="left", suffix="_new")
    for c in cols:
        # eq_missing: 片側だけ null の行も不一致として数える（== だと null になり all() が無視する）
        if not joined[c].eq_missing(joined[f"{c}_new"]).all():
            raise SystemExit(f"v1 preview reproduction failed on {c}")
    return saved.height // 6


def v1_frame(previews, end):
    """live_model.build_dataset() と同じ結合・検査（対象日だけ）。"""
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").filter(
        (pl.col("date") >= START) & (pl.col("date") <= end)).with_columns(
        class_ord=pl.col("class").replace_strict(v1.baseline.CLASS_ORD))
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet").filter(pl.col("date") <= end)
    historic = results.with_columns(st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64))
    rh, vb = racer_history(historic), venue_boat_history(historic)
    df = entries.join(previews, on=BK, validate="1:1")
    df = df.join(rh, on=["racer_id", "date"], validate="m:1").join(vb, on=["jcd", "boat", "date"], validate="m:1")
    labels = results.select(*BK, "racer_id", pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8).alias("rank"))
    df = df.join(labels, on=BK + ["racer_id"], validate="1:1")
    accepted = []
    for key, records in groupby(df.sort(BK).iter_rows(named=True), key=lambda r: tuple(r[c] for c in KEY)):
        records = list(records)
        try:
            v1.validate_records(records)
            if any(r["rank"] is None for r in records) or sorted(r["rank"] for r in records) != list(range(1, 7)):
                raise ValueError("Not six distinct finishers")
            accepted.append(key)
        except (ValueError, TypeError):
            pass
    return v1.derive(df.join(pl.DataFrame(accepted, schema=KEY, orient="row"), on=KEY).sort(BK))


def losses_by_key(df, p1, joint):
    return df.select(KEY).unique(maintain_order=True).with_columns(
        pl.Series("loss", v1.losses(df, p1, joint).tolist()))


def main():
    for path, sha in FROZEN.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"Frozen artifact changed: {path}")
    if OUT.exists():
        raise SystemExit("Result already recorded; refusing overwrite")
    end = max(p.stem for p in (ROOT / "data/raw/boatracecsv/od2").glob("*.csv"))
    results_end = pl.read_parquet(ROOT / "data/parsed/results.parquet")["date"].max()
    end = min(end, results_end)
    if START <= HOLDOUT_END:
        raise SystemExit("Prospective period overlaps the sealed holdout")
    report = dict(period=[START, end], evaluated_at=datetime.now(timezone.utc).isoformat(),
                  v1_reproduction_races=check_v1_reproduction(), frozen_hashes_verified=True)

    frames = {k: raw_previews(k, START, end) for k in ("tkz", "stt", "sui", "od2")}
    # 展示タイム: 直前CSV と K文書の一致（v2 は学習も評価も K文書の値を使う）
    tk = frames["tkz"].filter((pl.col("minutes_before_scheduled_deadline") > 0)
                              & (pl.col("minutes_before_scheduled_deadline") <= 60))
    tenji_pre = pl.concat([tk.select(*KEY, pl.lit(b).cast(pl.Int64).alias("boat"),
                                     pl.col(f"艇{b}_展示タイム").cast(pl.Float64, strict=False).alias("pre"))
                           for b in range(1, 7)])

    # v1（凍結）: 直前情報3ソースが揃うレースだけ
    f1 = v1_frame(v1_previews({k: frames[k] for k in ("tkz", "stt", "sui")}), end)
    model1 = v1.load_model(v1.OUT / "preview_model.pkl")
    p1_v1, j_v1 = model1.predict(f1)
    # v2（凍結）: 正常レース全体から、展示タイムが締切前60分以内に出ていたレースに絞る
    f2 = v2.build_frame(end=end).filter(pl.col("date") >= START)
    pre_ok = (f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="left")
              .group_by(KEY).agg(ok=pl.col("pre").eq_missing(pl.col("tenji")).all(), n=pl.col("pre").is_not_null().sum())
              .filter(pl.col("ok") & (pl.col("n") == 6)).select(KEY))
    mismatch = f2.select(*BK, "tenji").join(tenji_pre, on=BK, how="inner").filter(pl.col("pre") != pl.col("tenji")).height
    f2 = f2.join(pre_ok, on=KEY).sort(BK)
    model2 = v2.load_model(v2.OUT / "model.pkl")
    p1_v2, j_v2 = model2.predict(f2)
    report["races"] = dict(v1_predictable=f1.height // 6, v2_predictable=f2.height // 6,
                           tenji_preview_vs_k_mismatched_boats=mismatch)

    # Q1: 両方が予測できるレースで対応付き
    l1 = losses_by_key(f1, p1_v1, j_v1)
    l2 = losses_by_key(f2, p1_v2, j_v2)
    both = l1.join(l2, on=KEY, suffix="_v2").sort(KEY)
    a, b = np.array(both["loss"].to_list()), np.array(both["loss_v2"].to_list())
    report["Q1"] = dict(races=both.height, order=["1着", "2連単", "2連複"],
                        v1=a.mean(axis=0).tolist(), v2=b.mean(axis=0).tolist(),
                        v2_minus_v1=(b - a).mean(axis=0).tolist(),
                        day_bootstrap_95ci=v2.day_bootstrap(b - a, both["date"].to_numpy()))
    ci = report["Q1"]["day_bootstrap_95ci"]
    report["Q1"]["verdict"] = "v2 better" if ci[1][1] < 0 and ci[2][1] < 0 else "not established"

    # 市場: od2 板（締切前60分以内）× 通常払戻
    od = frames["od2"].filter((pl.col("minutes_before_scheduled_deadline") > 0)
                              & (pl.col("minutes_before_scheduled_deadline") <= 60))
    coef1 = json.loads((v1.OUT / "market_coefficients.json").read_text())
    coef2 = json.loads((v2.OUT / "market_coefficients.json").read_text())
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet").filter(pl.col("date") >= START)
    report["minutes_before_deadline_od2"] = np.quantile(
        od["minutes_before_scheduled_deadline"].to_numpy(), [.05, .5, .95]).tolist()
    for pool in m2.POOLS:
        pay = (payouts.filter(pl.col("bet_type") == m2.LABELS[pool]).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        out = {}
        for name, frame, joint in (("v2", f2, j_v2), ("v1", f1, j_v1)):
            keys = frame.select(KEY).unique(maintain_order=True)
            rank = frame["rank"].to_numpy().reshape(-1, 6)
            d = keys.with_row_index("i").join(od, on=KEY).join(pay, on=KEY).sort("i")
            o2t, o2f = m2.board_arrays(d)
            idx = d["i"].to_numpy()
            first, second = rank[idx].argmin(axis=1) + 1, np.argsort(rank[idx], axis=1)[:, 1] + 1
            win = [(x, y) if pool == "2t" else (min(x, y), max(x, y)) for x, y in zip(first, second)]
            normal = (d["n"].to_numpy() == 1) & np.array([c == f"{x}-{y}" for c, (x, y) in zip(d["combo"], win)])
            complete = np.isfinite(o2t).all(axis=1) & (np.isfinite(o2f).all(axis=1) if pool == "2f" else True)
            keep = normal & complete
            o2t, o2f, idx = o2t[keep], o2f[keep], idx[keep]
            y = np.array([m2.POOLS[pool].index(w) for w, k in zip(win, keep) if k])
            amount = d["amount"].to_numpy().astype(float)[keep]
            dates = d["date"].to_numpy()[keep]
            p = joint[idx] if pool == "2t" else m2.quinella_from_exacta(joint[idx])
            board = o2t if pool == "2t" else o2f
            r = np.arange(len(y))
            final_market = -np.log(np.clip((100 / amount) / m2.TAKE[pool], 1e-15, 1))
            if name == "v2":
                P = m2.distribution([np.log(np.clip(p, 1e-15, 1))] + m2.market_inputs(pool, o2t, o2f),
                                    coef2[pool]["combined"])
                settle = m2.settle_factor(m2.drift_design(pool, P, o2t, o2f), coef2[pool]["drift"])
            else:
                P = m1.distribution(p, m2.normalize_inverse(board), coef1[pool]["combined"])
                settle = None
            loss = -np.log(np.clip(P[r, y], 1e-15, 1))
            diff = loss - final_market
            ev_captured = P * board
            item = dict(races=int(len(y)), final_market_logloss=float(final_market.mean()),
                        combined_logloss=float(loss.mean()),
                        combined_minus_final_market=dict(mean=float(diff.mean()),
                                                         day_bootstrap_95ci=v2.day_bootstrap(diff[:, None], dates)[0]),
                        captured_market_logloss=float(-np.log(m2.normalize_inverse(board)[r, y]).mean()),
                        paper_ev_captured_gt_1=m2.paper(ev_captured > 1, y, amount, dates))
            if settle is not None:
                ev_settle = ev_captured * settle
                chosen = ev_captured > 1
                item["Q3"] = dict(bets=int(chosen.sum()),
                                  realized_roi=item["paper_ev_captured_gt_1"]["roi"],
                                  mean_ev_captured=float(ev_captured[chosen].mean()) if chosen.any() else None,
                                  mean_ev_settle=float(ev_settle[chosen].mean()) if chosen.any() else None)
                q3 = item["Q3"]
                if q3["realized_roi"] is not None:
                    q3["verdict"] = ("drift correction works" if abs(q3["realized_roi"] - q3["mean_ev_settle"])
                                     < abs(q3["realized_roi"] - q3["mean_ev_captured"]) else "not established")
                item["paper_ev_settle_gt_1"] = m2.paper(ev_settle > 1, y, amount, dates)
            out[name] = item
        report[pool] = out
        print(pool, json.dumps(out, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("period", "races", "Q1")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
