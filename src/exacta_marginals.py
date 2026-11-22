"""2連単の板を着順別に集約する市場合成の時系列実験。"""
import hashlib
import json

import numpy as np
import polars as pl

import live_market_v4 as v4
from live_market import day_ci, roi_ci

ROOT, m2 = v4.ROOT, v4.m2
PAIRS = np.asarray(m2.POOLS["2t"]) - 1
REPORT = ROOT / "prereg/exacta_marginals_result.json"


def features(p, odds):
    q = m2.normalize_inverse(odds)
    columns = [np.log(p), np.log(q)]
    for lanes in PAIRS.T:
        masks = np.eye(6)[lanes]
        # Multiple tickets express a view about the same boat finishing in this position.
        ratio = (q @ masks) / (p @ masks)
        columns.append(np.log(ratio[:, lanes]))
    return columns


def load(lo, hi, filename):
    freeze = json.loads((ROOT / "prereg/live_v4_freeze.json").read_text())["sha256"]
    path = ROOT / "data/models/live_v4a" / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != freeze[str(path.relative_to(ROOT))]:
        raise ValueError("OOS file differs from frozen predictions")
    keys = pl.read_parquet(path, columns=[*v4.KEY, "model_trained_before"])
    if (keys.select(v4.KEY).is_duplicated().any() or keys["date"].min() < lo
            or keys["date"].max() > hi or (keys["model_trained_before"] > keys["date"]).any()):
        raise ValueError("Invalid OOS split")

    # Match the exhibition input actually available before the scheduled deadline.
    tk = v4.ep2.raw_previews("tkz", lo, hi).filter(
        pl.col("minutes_before_scheduled_deadline").is_between(0, 60, closed="right"))
    actual = pl.read_parquet(ROOT / "data/parsed/results.parquet", columns=[*v4.KEY, "boat", "tenji"])
    preview = pl.concat([tk.select(*v4.KEY, pl.lit(i).alias("boat"),
                                   pl.col(f"艇{i}_展示タイム").cast(pl.Float64, strict=False).alias("pre"))
                         for i in range(1, 7)])
    valid = (preview.join(actual, on=[*v4.KEY, "boat"]).group_by(v4.KEY)
             .agg(ok=(pl.col("pre").is_not_null() & (pl.col("pre") == pl.col("tenji"))).all(),
                  n=pl.len()).filter(pl.col("ok") & (pl.col("n") == 6)))
    od = v4.boards(lo, hi).join(valid.select(v4.KEY), on=v4.KEY)
    return v4.m3.load_dataset(path, od)["2t"]


def fit(d, marginal):
    x = features(d["p"], d["o2t"])
    if not marginal:
        x = x[:2]
    coef = m2.fit(x, d["y"])
    prob = m2.distribution(x, coef)
    design = m2.drift_design("2t", prob, d["o2t"], None)
    return dict(combined=coef, drift=m2.fit_drift(design, d["o2t"], d["y"], d["amount"]))


def evaluate(d, coef):
    x = features(d["p"], d["o2t"])[:len(coef["combined"])]
    p = m2.distribution(x, coef["combined"])
    y = d["y"]
    row = np.arange(len(y))
    loss = -np.log(p[row, y])
    ev = p * d["o2t"] * m2.settle_factor(m2.drift_design("2t", p, d["o2t"], None), coef["drift"])
    return dict(log_loss=loss.mean(), top1=(p.argmax(axis=1) == y).mean(), **paper_results(d, ev)), loss


def paper_results(d, ev):
    bets = ev > 1
    stakes = bets.sum(axis=1) * 100
    returns = bets[np.arange(len(bets)), d["y"]] * d["amount"]
    return dict(bets=int(bets.sum()), roi=returns.sum() / stakes.sum(),
                roi_ci=roi_ci(stakes, returns, d["dates"]))


def paired_results(delta, dates):
    result = {}
    for name, mask in [("all", np.ones(len(delta), bool)), ("through_oct09", dates <= "20261009"),
                       ("from_oct10", dates >= "20261010")]:
        result[name] = dict(n=int(mask.sum()), delta=delta[mask].mean(), ci=day_ci(delta[mask], dates[mask]))
    return result


def experiment():
    tr = load("20260719", "20260909", "oos_predictions.parquet")
    te = load("20260910", "20261112", "oos_20260910_20261112.parquet")
    report = dict(train=len(tr["y"]), test=len(te["y"]), calibration={})
    losses = []
    for name, marginal in [("v4", False), ("marginals", True)]:
        coef = fit(tr, marginal)
        report["calibration"][name] = coef
        report[name], loss = evaluate(te, coef)
        losses.append(loss)
    delta = losses[1] - losses[0]
    report["paired"] = paired_results(delta, te["dates"])
    print(json.dumps(report, indent=2))
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    experiment()
