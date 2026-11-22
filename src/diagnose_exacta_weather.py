"""締切前の天候・風速で前回の2連単検証を層別する。"""
import json

import numpy as np
import polars as pl

from exacta_marginals import ROOT, load, m2, v4
from exacta_settlement import design, implied_votes
from live_market import day_ci, roi_ci

LO, HI = "20260910", "20261112"
WEATHER = {"1": "晴", "2": "曇り", "3": "雨"}


def main():
    d = load(LO, HI, "oos_20260910_20261112.parquet")
    original = json.loads((ROOT / "prereg/exacta_settlement_result.json").read_text())
    coef = json.loads((ROOT / "data/models/live_v4a/market_coefficients.json").read_text())["2t"]
    odds = d["o2t"]
    prob = m2.distribution([np.log(d["p"]), np.log(m2.normalize_inverse(odds))], coef["combined"])
    votes = np.array([implied_votes(board) for board in odds])
    inputs = design(prob, odds, votes, original["median"])
    drift = original["models"]["liquidity"]["drift"]
    bets = prob * odds * m2.settle_factor(inputs, drift) > 1
    y = d["y"]
    race = np.arange(len(y))
    stakes = bets.sum(axis=1) * 100
    hits = bets[race, y]
    returns = hits * d["amount"]
    baseline = original["models"]["liquidity"]
    if (len(y) != original["test"] or bets.sum() != baseline["bets"]
            or not np.isclose(returns.sum() / stakes.sum(), baseline["roi"])):
        raise ValueError("Race set or paper rule differs from the settlement evaluation")
    loss = -np.log(prob[race, y])
    market_loss = -np.log(np.clip((100 / d["amount"]) / m2.TAKE["2t"], 1e-15, 1))
    correct = prob.argmax(axis=1) == y

    sui = v4.ep2.raw_previews("sui", LO, HI).filter(
        pl.col("minutes_before_scheduled_deadline").is_between(0, 60, closed="right"))
    # Only use weather from the same snapshot as the archived odds.
    boards = v4.boards(LO, HI).select(*v4.KEY, "取得日時")
    weather = sui.join(boards, on=[*v4.KEY, "取得日時"]).select(
        *v4.KEY, "天候", wind=pl.col("風速(m)").cast(pl.Float64, strict=False))
    aligned = d["keys"].join(weather, on=v4.KEY, how="left", maintain_order="left", validate="1:1")
    wind, code = aligned["wind"].to_numpy(), aligned["天候"].to_numpy()
    valid = np.isin(code, list(WEATHER)) & np.isfinite(wind) & (wind >= 0)
    conditions = [(label, code == value) for value, label in WEATHER.items()]
    bands = [("0–2 m/s", (wind >= 0) & (wind < 3)), ("3–5 m/s", (wind >= 3) & (wind < 6)),
             ("6 m/s以上", wind >= 6)]
    groups = [("全条件", valid), *conditions, *bands]
    groups.extend((f"{name} × {band}", a & b) for name, a in conditions for band, b in bands)
    rows = []
    for label, mask in groups:
        mask = valid & mask
        if not mask.any():
            continue
        dates = d["dates"][mask]
        diff = (loss - market_loss)[mask]
        lo, hi = day_ci(diff, dates)
        stake = stakes[mask].sum()
        ci = roi_ci(stakes[mask], returns[mask], dates) if stake else None
        rows.append(dict(condition=label, races=int(mask.sum()), days=len(np.unique(dates)),
                         top1=correct[mask].mean(), mean_top1_probability=prob[mask].max(axis=1).mean(),
                         log_loss=loss[mask].mean(), minus_final_market=diff.mean(), diff_lo=lo, diff_hi=hi,
                         tickets=int(stake // 100), winning_tickets=int(hits[mask].sum()),
                         largest_payout_yen=int(returns[mask].max()),
                         roi=returns[mask].sum()/stake if stake else None,
                         roi_lo=ci[0] if ci else None, roi_hi=ci[1] if ci else None))
    table = pl.DataFrame(rows)
    table.write_csv(ROOT / "prereg/exacta_weather.csv")
    print(f"Weather and wind available for {valid.sum()}/{len(y)} races")


if __name__ == "__main__":
    main()
