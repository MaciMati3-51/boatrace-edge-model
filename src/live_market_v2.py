"""v2 の市場合成・締切までのオッズ変動補正・紙上候補。

v1（live_market.py）との違い:

1. 2連複の市場確率に、2連複板に加えて2連単板から作った含意確率も入れる。締切9分前の2連複プールは
   薄く（0倍セル 6.3%）、2連複板そのものより2連単板から作った確率のほうが当たる
   （log-loss 2.126 vs 2.074、3,768レース）。
2. 期待値を「撮影時オッズ」ではなく「確定オッズの期待値」で出す。払戻は確定オッズで決まる。
   的中目の log(確定倍率/撮影オッズ) はモデルと市場の乖離 log(P/q) に対し傾き約 −0.7 で、
   締切までに入る資金が乖離の7割を消す。撮影時オッズの EV>1 はモデル上の回収率 132〜143% に対し
   実現 79〜92% だった。v2 の紙上候補は確定オッズ期待値 ev_settle > 1。
3. 板の検算。Σ(1/オッズ) は控除率で決まり、od2 の完全板 10,346枚で 1.333〜1.45 を外れたのは1枚。
   外れたら転記ミスを疑って止める（1.0倍の元返しがある板は下限を免除）。

学習: .venv/bin/python src/live_market_v2.py train   （live_v2.py train の後）
予測: .venv/bin/python src/live_market_v2.py predict --input race.csv --odds odds.csv --out prereg/live_v2_YYYYMMDD_JJ_RR
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp

import live_market as market_v1
import live_v2 as fund

ROOT = fund.ROOT
KEY = fund.KEY
OUT = fund.OUT
POOLS = market_v1.POOLS  # 1始まり。2t は (1,2),(1,3)…、2f は (1,2),(1,3)…(5,6)
LABELS = market_v1.LABELS
CALIBRATION, TEST = ("20260719", "20260819"), ("20260820", "20260909")
V1_TEST_START = "20260901"  # v1 の合成器評価期間。比較用に同じ区間も出す
TAKE = {"2t": 1.3407, "2f": 1.345}  # od2 完全板の Σ(1/オッズ) 中央値。確定市場の確率に使う
SUM_BOUNDS = (1.333, 1.45)
EXACTA_INDEX = {pair: i for i, pair in enumerate(POOLS["2t"])}


def normalize_inverse(odds):
    q = 1 / odds
    return q / q.sum(axis=-1, keepdims=True)


def quinella_from_exacta(p):
    """2連単30通り（POOLS['2t'] 順）→ 2連複15通り（POOLS['2f'] 順）。"""
    return np.stack([p[..., EXACTA_INDEX[(a, b)]] + p[..., EXACTA_INDEX[(b, a)]]
                     for a, b in POOLS["2f"]], axis=-1)


def market_inputs(pool, o2t, o2f):
    """券種ごとの市場側の入力（log確率のリスト）。2連複は2連単板からの含意も使う。"""
    q2t = normalize_inverse(o2t)
    if pool == "2t":
        return [np.log(q2t)]
    return [np.log(normalize_inverse(o2f)), np.log(quinella_from_exacta(q2t))]


def distribution(features, coef):
    z = np.stack(features, axis=-1) @ np.asarray(coef)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def fit(features, y):
    """条件付きロジット（レース内 softmax）。係数は 0〜5。v1 の fit を多入力にしたもの。"""
    x = np.stack(features, axis=-1)
    r = np.arange(len(y))

    def objective(coef):
        z = x @ coef
        logprob = z - logsumexp(z, axis=1, keepdims=True)
        residual = np.exp(logprob)
        residual[r, y] -= 1
        return -logprob[r, y].mean(), np.einsum("ij,ijk->k", residual, x) / len(y)

    result = minimize(objective, np.full(x.shape[-1], .5), jac=True, method="L-BFGS-B",
                      bounds=[(0, 5)] * x.shape[-1])
    if not result.success:
        raise ValueError(f"Calibration failed: {result.message}")
    return result.x.tolist()


def drift_design(pool, p_final, o2t, o2f):
    """的中した場合の log(確定倍率/撮影オッズ) を説明する列。全候補ぶん (レース, 目, 列)。"""
    q2t = normalize_inverse(o2t)
    if pool == "2t":
        q = q2t
        cols = [np.ones_like(q), np.log(p_final / q), np.log(q)]
    else:
        q = normalize_inverse(o2f)
        # 薄い2連複プールは確定までに2連単側の値段へ寄る
        cols = [np.ones_like(q), np.log(p_final / q), np.log(q), np.log(quinella_from_exacta(q2t) / q)]
    return np.stack(cols, axis=-1)


def fit_drift(design, odds, y, amount):
    """的中目だけで最小二乗。期待値の戻しは正規分布を仮定せず残差の smearing（Duan）で行う。"""
    r = np.arange(len(y))
    x = design[r, y]
    target = np.log(amount / 100 / odds[r, y])
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    smear = float(np.mean(np.exp(target - x @ beta)))
    return dict(beta=beta.tolist(), smearing=smear, winners=int(len(y)),
                residual_sd=float(np.std(target - x @ beta)))


def settle_factor(design, drift):
    return np.exp(design @ np.asarray(drift["beta"])) * drift["smearing"]


def board_arrays(odds_frame):
    """od2 の横持ち板 → (キー, 2連単30, 2連複15)。0倍・欠損は NaN。"""
    o2t = odds_frame.select([pl.col(f"2連単_{a}-{b}").cast(pl.Float64, strict=False)
                             for a, b in POOLS["2t"]]).to_numpy()
    o2f = odds_frame.select([pl.col(f"2連複_{a}={b}").cast(pl.Float64, strict=False)
                             for a, b in POOLS["2f"]]).to_numpy()
    o2t[~(np.nan_to_num(o2t) >= 1)] = np.nan
    o2f[~(np.nan_to_num(o2f) >= 1)] = np.nan
    return o2t, o2f


def load_dataset():
    """v2 の時系列外予測 × od2 板 × 通常払戻。締切前60分以内の板、正常6艇完走だけ。"""
    meta = json.loads((OUT / "metadata.json").read_text())
    if hashlib.sha256((OUT / "oos_predictions.parquet").read_bytes()).hexdigest() != meta["oos_sha256"]:
        raise ValueError("OOS predictions changed after live_v2 training")
    oos = pl.read_parquet(OUT / "oos_predictions.parquet")
    od = pl.read_parquet(ROOT / "data/parsed/boatracecsv/od2.parquet").filter(
        (pl.col("minutes_before_scheduled_deadline") > 0) & (pl.col("minutes_before_scheduled_deadline") <= 60))
    if od.unique(KEY).height != od.height:
        raise ValueError("Duplicate market records")
    d = oos.join(od, on=KEY, how="inner").sort(KEY)
    o2t, o2f = board_arrays(d)
    joint = np.array(d["joint"].to_list())
    rank = np.array(d["rank"].to_list())
    first, second = rank.argmin(axis=1) + 1, np.argsort(rank, axis=1)[:, 1] + 1
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet").filter(pl.col("date") >= CALIBRATION[0])
    data = {}
    for pool in POOLS:
        pay = (payouts.filter(pl.col("bet_type") == LABELS[pool]).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        pay = d.select(KEY).join(pay, on=KEY, how="left", maintain_order="left")
        winners = [(a, b) if pool == "2t" else (min(a, b), max(a, b)) for a, b in zip(first, second)]
        normal = ((pay["n"].to_numpy() == 1)
                  & np.array([c == f"{a}-{b}" for c, (a, b) in zip(pay["combo"].to_list(), winners)])
                  & (np.nan_to_num(pay["amount"].to_numpy().astype(float)) > 0))
        complete = np.isfinite(o2t).all(axis=1) & (np.isfinite(o2f).all(axis=1) if pool == "2f" else True)
        keep = normal & complete
        p = joint if pool == "2t" else quinella_from_exacta(joint)
        data[pool] = dict(dates=d["date"].to_numpy()[keep], keys=d.select(KEY).filter(pl.Series(keep)),
                          p=p[keep], o2t=o2t[keep], o2f=o2f[keep],
                          y=np.array([POOLS[pool].index(w) for w in winners])[keep],
                          amount=pay["amount"].to_numpy().astype(float)[keep],
                          counts=dict(boards=int(d.height), no_normal_payout=int((~normal).sum()),
                                      incomplete_board=int((normal & ~complete).sum()), used=int(keep.sum())))
    return data


def paper(bets, y, amount, dates):
    """各目100円。回収率は日単位ブートストラップの95%区間つき。"""
    r = np.arange(len(y))
    stakes = bets.sum(axis=1) * 100.
    returns = bets[r, y] * amount
    unique = np.unique(dates)
    s = np.array([stakes[dates == d].sum() for d in unique])
    ret = np.array([returns[dates == d].sum() for d in unique])
    picks = np.random.default_rng(0).integers(0, len(unique), (2000, len(unique)))
    den, num = s[picks].sum(axis=1), ret[picks].sum(axis=1)
    ok = den > 0
    return dict(bets=int(bets.sum()), stake_jpy=int(stakes.sum()), return_jpy=int(returns.sum()),
                roi=float(returns.sum() / stakes.sum()) if stakes.sum() else None,
                day_bootstrap_roi_95ci=np.quantile(num[ok] / den[ok], [.025, .975]).tolist() if ok.any() else None)


def evaluate(pool, d, calib, test):
    y, r = d["y"], np.arange(len(d["y"]))
    market = market_inputs(pool, d["o2t"], d["o2f"])
    board = d["o2t"] if pool == "2t" else d["o2f"]
    log_p = np.log(np.clip(d["p"], 1e-15, 1))
    methods = {"fundamental": ([log_p], [1.])}
    methods["market"] = (market[:1], [1.])
    methods["market_power"] = (market[:1], fit([market[0][calib]], y[calib]))
    if pool == "2f":
        methods["market_mix"] = (market, fit([m[calib] for m in market], y[calib]))
    features = [log_p] + market
    coef = fit([f[calib] for f in features], y[calib])
    methods["combined"] = (features, coef)
    combined = distribution(features, coef)
    design = drift_design(pool, combined, d["o2t"], d["o2f"])
    drift = fit_drift(design[calib], board[calib], y[calib], d["amount"][calib])
    settle = settle_factor(design, drift)
    final_market = -np.log(np.clip((100 / d["amount"]) / TAKE[pool], 1e-15, 1))
    report = dict(counts=d["counts"], calibration_races=int(calib.sum()), test_races=int(test.sum()),
                  drift=drift, methods={})
    for window, mask in (("test", test), ("v1_test_window", test & (d["dates"] >= V1_TEST_START))):
        dates = d["dates"][mask]
        out = dict(races=int(mask.sum()), final_market_logloss=float(final_market[mask].mean()))
        for name, (feats, c) in methods.items():
            prob = distribution([f[mask] for f in feats], c) if name != "fundamental" else d["p"][mask]
            loss = -np.log(np.clip(prob[np.arange(mask.sum()), y[mask]], 1e-15, 1))
            diff = loss - final_market[mask]
            ev_captured = prob * board[mask]
            item = dict(coefficients=c, logloss=float(loss.mean()),
                        minus_final_market=dict(mean=float(diff.mean()),
                                                day_bootstrap_95ci=fund.day_bootstrap(diff[:, None], dates)[0]),
                        paper_ev_captured_gt_1=paper(ev_captured > 1, y[mask], d["amount"][mask], dates))
            if name == "combined":
                ev_settle = ev_captured * settle[mask]
                item["paper_ev_settle_gt_1"] = paper(ev_settle > 1, y[mask], d["amount"][mask], dates)
                chosen = ev_captured > 1
                item["model_expected_roi_of_ev_captured_bets"] = float(ev_captured[chosen].mean()) if chosen.any() else None
                chosen = ev_settle > 1
                item["model_expected_roi_of_ev_settle_bets"] = float(ev_settle[chosen].mean()) if chosen.any() else None
            out[name] = item
        report["methods"][window] = out
    return report


def train():
    data = load_dataset()
    report = dict(calibration_period=list(CALIBRATION), test_period=list(TEST),
                  status="exploratory historical comparison; normal six-finisher races; boards captured about 9 minutes before the deadline",
                  final_market="probability of the winning combination from the settled payout: (100/payout)/TAKE",
                  bet_rules={"ev_captured": "P_final x captured odds > 1 (v1 rule)",
                             "ev_settle": "P_final x captured odds x E[settled/captured | win] > 1 (v2 rule)"},
                  pools={})
    saved = {}
    for pool, d in data.items():
        calib = (d["dates"] >= CALIBRATION[0]) & (d["dates"] <= CALIBRATION[1])
        test = (d["dates"] >= TEST[0]) & (d["dates"] <= TEST[1])
        report["pools"][pool] = evaluate(pool, d, calib, test)
        print(pool, json.dumps(report["pools"][pool]["methods"]["test"]["combined"], ensure_ascii=False), flush=True)
        # 保存用: 全期間（7/19-9/9）で再推定
        everything = calib | test
        market = market_inputs(pool, d["o2t"][everything], d["o2f"][everything])
        features = [np.log(np.clip(d["p"][everything], 1e-15, 1))] + market
        coef = fit(features, d["y"][everything])
        combined = distribution(features, coef)
        board = (d["o2t"] if pool == "2t" else d["o2f"])[everything]
        drift = fit_drift(drift_design(pool, combined, d["o2t"][everything], d["o2f"][everything]),
                          board, d["y"][everything], d["amount"][everything])
        saved[pool] = dict(combined=coef, drift=drift, races=int(everything.sum()),
                           inputs=["log p_fundamental"] + (["log q_2t"] if pool == "2t" else ["log q_2f", "log q_2f_from_2t"]))
    saved.update(fundamental_model_sha256=json.loads((OUT / "metadata.json").read_text())["model_sha256"],
                 fitted_on=[CALIBRATION[0], TEST[1]], generated_at=datetime.now(timezone.utc).isoformat(),
                 note="Coefficients fit on live_v2 out-of-sample predictions; the saved fundamental model is refit through 20260909.")
    (OUT / "market_coefficients.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "prereg/live_market_v2_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def validate_odds(df):
    df = market_v1.validate_odds(df)
    for group in df.partition_by(KEY + ["pool"]):
        odds = group["odds"].to_numpy().astype(float)
        total = (1 / odds).sum()
        low, high = SUM_BOUNDS
        if total > high or (total < low and not (odds == 1).any()):
            raise ValueError(f"Sum of 1/odds {total:.4f} outside {SUM_BOUNDS} for {group['pool'][0]}: "
                             "check the transcription")
    return df


def predict(input_path, odds_path, out):
    inputs = fund.validate_input(pl.read_csv(input_path, schema_overrides={"date": pl.String}))
    odds = validate_odds(pl.read_csv(odds_path, schema_overrides={"date": pl.String}))
    input_keys = inputs.select(KEY + ["deadline_at"]).unique()
    if odds.select(KEY + ["deadline_at"]).unique().join(input_keys, on=KEY + ["deadline_at"], how="anti").height:
        raise ValueError("Odds race/deadline does not match exhibition input")
    coef = json.loads((OUT / "market_coefficients.json").read_text())
    sha = hashlib.sha256((OUT / "model.pkl").read_bytes()).hexdigest()
    if sha != coef["fundamental_model_sha256"]:
        raise ValueError("Fundamental model changed; recalibrate market coefficients")
    if out.exists():
        raise ValueError("Output directory already exists")
    boards = {}
    for group in odds.partition_by(KEY + ["pool"]):
        row = group.row(0, named=True)
        lookup = {(a, b): o for a, b, o in zip(group["first"], group["second"], group["odds"])}
        boards[(row["date"], row["jcd"], row["race_no"], row["pool"])] = (
            np.array([lookup[pair] for pair in POOLS[row["pool"]]], dtype=float), row)
    for key in inputs.select(KEY).unique().iter_rows():
        if (*key, "2t") not in boards:
            raise ValueError("2連単 board is required (the 2連複 market also uses it)")
    out.mkdir(parents=True)
    fund.predict(input_path, out / "fundamental.csv")
    p = pl.read_csv(out / "fundamental.csv", schema_overrides={"date": pl.String})
    records = []
    for key in inputs.select(KEY).unique().sort(KEY).iter_rows():
        pp = p.filter(pl.all_horizontal([pl.col(c) == v for c, v in zip(KEY, key)])).sort("first", "second")
        p2t = pp["p_2t"].to_numpy()[None]
        o2t, _ = boards[(*key, "2t")]
        for pool in POOLS:
            if (*key, pool) not in boards:
                continue
            o2f = boards[(*key, "2f")][0][None] if pool == "2f" else np.full((1, 15), np.nan)
            p_f = p2t if pool == "2t" else quinella_from_exacta(p2t)
            market = market_inputs(pool, o2t[None], o2f)
            final = distribution([np.log(np.clip(p_f, 1e-15, 1))] + market, coef[pool]["combined"])
            board = o2t[None] if pool == "2t" else o2f
            ev_captured = final * board
            ev_settle = ev_captured * settle_factor(drift_design(pool, final, o2t[None], o2f), coef[pool]["drift"])
            meta = boards[(*key, pool)][1]
            for j, (a, b) in enumerate(POOLS[pool]):
                records.append(dict(date=key[0], jcd=key[1], race_no=key[2], pool=pool, first=a, second=b,
                                    odds=float(board[0, j]), captured_at=meta["captured_at"],
                                    deadline_at=meta["deadline_at"], p_fundamental=float(p_f[0, j]),
                                    p_market=float(np.exp(market[0][0, j])),
                                    p_market_from_2t=float(np.exp(market[1][0, j])) if pool == "2f" else None,
                                    p_final=float(final[0, j]), ev_captured=float(ev_captured[0, j]),
                                    ev_settle=float(ev_settle[0, j]),
                                    paper_candidate=bool(ev_settle[0, j] > 1),
                                    v1_rule_candidate=bool(ev_captured[0, j] > 1)))
    pl.DataFrame(records).write_csv(out / "final.csv")
    (out / "metadata.json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v2", model_sha256=sha,
        calibration=coef, odds_sha256=hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        note="paper_candidate uses ev_settle (expected settled odds). Research output; profitability not established."),
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("--input", type=Path)
    ap.add_argument("--odds", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "train":
        train()
    elif args.input and args.odds and args.out:
        predict(args.input, args.odds, args.out)
    else:
        ap.error("predict requires --input, --odds and --out")
