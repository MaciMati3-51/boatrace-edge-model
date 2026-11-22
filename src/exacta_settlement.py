"""2連単の的中時の確定倍率を、撮影時の推定票数で補正する。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from exacta_liquidity import implied_votes
from exacta_marginals import ROOT, load, m2, paired_results, paper_results

CALIBRATION = ROOT / "prereg/exacta_settlement_coefficients.json"


def design(prob, odds, votes, median):
    base = m2.drift_design("2t", prob, odds, None)
    n = np.where(np.isfinite(votes), votes, median)
    weight = (n / (n + median))[:, None, None]
    return np.concatenate([base, base * weight], axis=-1)


def compare():
    train = load("20260719", "20260909", "oos_predictions.parquet")
    test = load("20260910", "20261112", "oos_20260910_20261112.parquet")
    # Historical operational baseline, including its original calibration sample.
    coef = json.loads((ROOT / "data/models/live_v4a/market_coefficients.json").read_text())
    votes, inputs, probs = [], [], []
    for d in (train, test):
        odds = d["o2t"]
        p = m2.distribution([np.log(d["p"]), np.log(m2.normalize_inverse(odds))], coef["2t"]["combined"])
        probs.append(p)
        votes.append(np.array([implied_votes(board) for board in odds]))
        inputs.append(design(p, odds, votes[-1], np.nanmedian(votes[0])))
    y, odds, dates = test["y"], test["o2t"], test["dates"]
    r = np.arange(len(y))
    report = dict(train=len(train["y"]), test=len(y), median=float(np.nanmedian(votes[0])), models={})
    losses = {}
    for name, ncols in [("v4_refit", 3), ("liquidity", 6), ("v4_frozen", 3)]:
        drift = coef["2t"]["drift"] if name == "v4_frozen" else m2.fit_drift(
            inputs[0][:, :, :ncols], train["o2t"], train["y"], train["amount"])
        expected = odds * m2.settle_factor(inputs[1][:, :, :ncols], drift)
        losses[name] = np.log(expected[r, y] / (test["amount"] / 100)) ** 2
        report["models"][name] = dict(drift=drift, mse=losses[name].mean(), **paper_results(test, expected * probs[1]))
    report["paired"] = {base: paired_results(losses["liquidity"] - losses[base], dates)
                        for base in ["v4_refit", "v4_frozen"]}
    (ROOT / "prereg/exacta_settlement_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def fit_candidate():
    parts = [load("20260719", "20260909", "oos_predictions.parquet"),
             load("20260910", "20261112", "oos_20260910_20261112.parquet")]
    d = {key: np.concatenate([part[key] for part in parts]) for key in ["p", "o2t", "y", "amount", "dates"]}
    coef_path = ROOT / "data/models/live_v4/market_coefficients.json"
    v4 = json.loads(coef_path.read_text())
    odds = d["o2t"]
    prob = m2.distribution([np.log(d["p"]), np.log(m2.normalize_inverse(odds))], v4["2t"]["combined"])
    votes = np.array([implied_votes(board) for board in odds])
    median = float(np.nanmedian(votes))
    drift = m2.fit_drift(design(prob, odds, votes, median), odds, d["y"], d["amount"])
    record = dict(model_sha256=v4["fundamental_model_sha256"], combined=v4["2t"]["combined"],
                  fitted_through=str(d["dates"].max()), races=len(votes), median=median, drift=drift)
    CALIBRATION.write_text(json.dumps(record, indent=2) + "\n")


def score(directory, output):
    coef = json.loads(CALIBRATION.read_text())
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["model_sha256"] != coef["model_sha256"] or meta["calibration"]["2t"]["combined"] != coef["combined"]:
        raise ValueError("Input must use the matching v4 model and market calibration")
    table = pl.read_csv(directory / "final.csv", schema_overrides={"date": pl.String}).filter(pl.col("pool") == "2t")
    table = m2.validate_odds(table).sort(m2.KEY + ["first", "second"])
    if table["date"].min() <= coef["fitted_through"]:
        raise ValueError("Scoring date overlaps model fitting")
    odds = table["odds"].to_numpy().reshape(-1, 30)
    p = table["p_final"].to_numpy().reshape(-1, 30)
    if not (np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(axis=1), 1)):
        raise ValueError("Invalid v4 probabilities")
    votes = np.array([implied_votes(board) for board in odds])
    expected = odds * m2.settle_factor(design(p, odds, votes, coef["median"]), coef["drift"])
    ev = expected * p
    result = table.rename({"ev_settle": "v4_ev_settle", "paper_candidate": "v4_paper_candidate"}).with_columns(
        estimated_votes=pl.Series(np.repeat(votes, 30)).fill_nan(None),
        expected_settled_odds=pl.Series(expected.ravel()), ev_settle=pl.Series(ev.ravel()),
        paper_candidate=pl.Series(ev.ravel() > 1),
        calibration_sha256=pl.lit(hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()))
    with output.open("x") as f:
        result.write_csv(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("fit")
    predict = sub.add_parser("score")
    predict.add_argument("--v4-dir", required=True, type=Path)
    predict.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "score":
        score(args.v4_dir, args.out)
    elif args.command == "fit":
        fit_candidate()
    else:
        compare()
