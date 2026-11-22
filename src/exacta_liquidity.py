"""Estimate a board's relative liquidity from its odds grid and calibrate exacta probabilities."""
import json

import numpy as np

from exacta_marginals import ROOT, evaluate, fit, load, m2, paired_results, paper_results

REPORT = ROOT / "prereg/exacta_liquidity_result.json"


def implied_votes(odds):
    """Smallest feasible pool assuming floor(7.5 * total / combo_votes) odds ticks."""
    ticks = np.rint(odds * 10).astype(np.int64)
    # Caps, minimum payouts and nonstandard displays cannot use these rounding intervals.
    if ticks.min() <= 10 or ticks.max() >= 99990 or not np.allclose(ticks, 10 * odds):
        return np.nan
    # Enumerate the possible ticket count on the longest shot first. This bounds the
    # search while retaining the sparse pools where an estimate matters most.
    top = ticks.max()
    for count in range(1, 129):
        start = (2 * count * top + 14) // 15
        stop = (2 * count * (top + 1) + 14) // 15
        totals = np.arange(start, stop, dtype=np.int64)
        numer = 15 * totals[:, None]
        lower = numer // (2 * (ticks + 1)) + 1
        upper = numer // (2 * ticks)
        fits = (lower <= upper).all(axis=1) & (lower.sum(axis=1) <= totals) & (totals <= upper.sum(axis=1))
        if fits.any():
            return int(totals[fits][0])
    return np.nan


def liquidity_features(p, odds, votes, median):
    n = np.where(np.isfinite(votes), votes, median)
    thick = (n / (n + median))[:, None]
    disagreement = np.log(m2.normalize_inverse(odds) / p)
    return [np.log(p), (1 - thick) * disagreement, thick * disagreement]


def train(d, votes):
    median = float(np.nanmedian(votes))
    x = liquidity_features(d["p"], d["o2t"], votes, median)
    combined = m2.fit(x, d["y"])
    p = m2.distribution(x, combined)
    drift = m2.fit_drift(m2.drift_design("2t", p, d["o2t"], None), d["o2t"], d["y"], d["amount"])
    return dict(median=median, combined=combined, drift=drift)


def apply(p, odds, coef, votes=None):
    if votes is None:
        votes = np.array([implied_votes(board) for board in odds])
    features = liquidity_features(p, odds, votes, coef["median"])
    prob = m2.distribution(features, coef["combined"])
    ev = prob * odds * m2.settle_factor(m2.drift_design("2t", prob, odds, None), coef["drift"])
    return prob, ev


def experiment():
    tr = load("20260719", "20260909", "oos_predictions.parquet")
    te = load("20260910", "20261112", "oos_20260910_20261112.parquet")
    votes_tr, votes_te = [np.array([implied_votes(board) for board in d["o2t"]]) for d in (tr, te)]
    coef = train(tr, votes_tr)
    base, base_loss = evaluate(te, fit(tr, False))
    p, ev = apply(te["p"], te["o2t"], coef, votes_te)
    y, dates = te["y"], te["dates"]
    rows = np.arange(len(y))
    loss = -np.log(p[rows, y])
    report = dict(calibration=coef, train=len(votes_tr), test=len(y),
                  estimated_train=int(np.isfinite(votes_tr).sum()), estimated_test=int(np.isfinite(votes_te).sum()),
                  v4=base, candidate=dict(log_loss=loss.mean(), top1=(p.argmax(axis=1) == y).mean(),
                                         **paper_results(te, ev)),
                  paired=paired_results(loss - base_loss, dates))
    print(json.dumps(report, indent=2))
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    experiment()
