"""直前情報モデル v3 — 特徴量を約100列/艇に増やし、艇単位の条件付きロジットに替える。

v2 からの変更（数値は開発区間: 学習 〜20240630、検証 20240701〜20250718 の 54,319レース。封印期間は見ていない）:

1. 特徴量（v3_features.py）: 選手×枠の成績・決まり手、直近90日/365日の成績、枠の有利不利を引いた残差、
   モーター使用期間内の成績、今節の成績、F からの日数など。v2 は選手通算5列＋会場×枠1列だった
2. モデル: 1艇1行で全艇共通の木を育て、レース内 softmax で確率にする（条件付きロジットの勾配ブースティング）。
   v2 は6艇の特徴を横に並べた6クラス分類で、枠ごとに別々の木を育てていた。同じことを枠の数だけ学び直すので
   データ効率が悪い。1着段は両方を幾何平均で混ぜる（艇単位 0.6 / 6クラス 0.4）
3. 2着段: 1着艇を所与として残り5艇の softmax（艇単位）。候補艇・1着艇・全艇の主要特徴を見る
4. 学習データ: 20190901〜（v2 は 20220901〜）。公式 B/K を 2019-07 まで遡って取得した

木の本数は「学習末尾90日を検証に回して早期停止 → その本数×1.1 で全体を学習し直す」で決める。

特徴量: .venv/bin/python src/v3_features.py           （entries/results を更新したら先に流す）
開発  : .venv/bin/python src/live_v3.py dev            （封印前だけで v2 と比較）
学習  : .venv/bin/python src/live_v3.py train
予測  : .venv/bin/python src/live_v3.py predict --input race.csv --out pred.csv
        入力CSVは v2 と同じ。ただし出走表は当日の公式 B（entries.parquet）から取るので、
        当日の B を取り込んでから実行する（モーター番号・ボート番号・レース種別が要る）。
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from holdout import HOLDOUT_START
import live_model as v1
import live_v2 as v2
import v3_features as feat

ROOT = Path(__file__).resolve().parents[1]
KEY, BK = feat.KEY, feat.BK
OUT = ROOT / "data/models/live_v3"
FEATURE_CACHE = ROOT / "data/features/v3.parquet"
PAIRS = v2.PAIRS
TRAIN_START, END = "20190901", "20260909"
OOS_FOLDS = v2.OOS_FOLDS
DEV = dict(train_before="20240701", valid_through="20250718")  # 封印 20250719〜 の手前
EARLY_STOP_DAYS, REFIT_FACTOR = 90, 1.1
BLEND_CL = 0.6  # 1着段: 艇単位 0.6 / 6クラス 0.4 の幾何平均
THREADS = 8

# 相対値（レース平均との差・レース内順位）を作る列
REL = ["nat_wr", "nat_2r", "class_ord", "r_win_all", "r_win_365", "r_top2_365", "r_pts_365", "r_rpts_365",
       "r_rwin_all", "r_rpts_all", "l_win", "l_top2", "l_top3", "r_st_365", "l_st", "m_rpts", "m_tj", "s_rpts",
       "motor_2r", "r_rpts_90"]
# 隣の艇・1着艇・全艇ぶんを持たせる主要列
MAIN = ["nat_wr", "class_ord", "r_win_all", "r_win_365", "r_rpts_365", "r_rwin_all", "l_win", "l_top2", "l_st",
        "r_st_365", "tenji_rel", "m_rpts", "s_rpts", "l_nige", "l_makuri", "l_sashi", "l_mzashi", "l_l1win",
        "r_cdiff", "l_moved"]
PARAMS = dict(learning_rate=0.03, lambda_l2=50.0, bagging_fraction=0.7, bagging_freq=1, verbose=-1,
              num_threads=THREADS, seed=0, deterministic=True, force_row_wise=True)
PARAMS_CL = dict(PARAMS, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.3)
PARAMS_FLAT = dict(PARAMS, objective="multiclass", num_class=6, num_leaves=15, min_data_in_leaf=300,
                   feature_fraction=0.2)


def group_softmax(scores, size):
    s = scores.reshape(-1, size)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def objective(size):
    """レース内 softmax の負の対数尤度。行は size 個ずつ同じレース。"""
    def fn(scores, data):
        y = data.get_label().reshape(-1, size)
        p = group_softmax(scores, size)
        return (p - y).ravel(), np.maximum(p * (1 - p), 1e-6).ravel()
    return fn


def metric(size):
    def fn(scores, data):
        y = data.get_label().reshape(-1, size)
        p = group_softmax(scores, size)
        return "logloss", float(-np.log(np.clip((p * y).sum(axis=1), 1e-15, 1)).mean()), False
    return fn


def load_frame(lo=TRAIN_START, hi=END):
    """正常レース（6艇完走）の特徴量。1艇1行、レース内は枠順。"""
    df = pl.read_parquet(FEATURE_CACHE).filter(
        pl.col("rank").is_not_null() & (pl.col("date") >= lo) & (pl.col("date") <= hi))
    # 出走表と結果で選手が違う艇（欠場の差し替え）があるレースは6行揃わないので落とす
    full = df.group_by(KEY).len().filter(pl.col("len") == 6).select(KEY)
    return df.join(full, on=KEY).sort(BK)


class Design:
    """特徴量フレーム → 学習・予測用の配列。"""

    def __init__(self, df, per_boat=None):
        df = df.with_columns(
            [(pl.col(c) - pl.col(c).mean().over(KEY)).alias(f"{c}_rel") for c in REL]
            + [pl.col(c).rank("average").over(KEY).alias(f"{c}_rk") for c in REL])
        self.per_boat = per_boat or feat.per_boat_columns(df)
        missing = [c for c in self.per_boat + feat.RACE_LEVEL if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing: {missing}")
        near = df
        for shift, tag in ((1, "in"), (-1, "out")):  # 1つ内側・1つ外側の艇
            near = near.with_columns([pl.col(c).shift(shift).over(KEY).alias(f"{c}_{tag}") for c in MAIN])
        near = near.with_columns([pl.col(c).max().over(KEY).alias(f"{c}_max") for c in MAIN[:10]])
        extra = [f"{c}_{t}" for c in MAIN for t in ("in", "out")] + [f"{c}_max" for c in MAIN[:10]]
        self.boat_names = ["boat"] + feat.RACE_LEVEL + self.per_boat + extra
        self.n = df.height // 6
        f32 = lambda frame, cols: frame.select(cols).to_numpy().astype(np.float32)
        self.boat = f32(near, self.boat_names).reshape(self.n, 6, -1)
        self.main = f32(df, MAIN).reshape(self.n, 6, -1)
        self.lineup = self.main.reshape(self.n, -1)
        self.lineup_names = [f"{c}_L{b}" for b in range(1, 7) for c in MAIN]
        race = f32(df, feat.RACE_LEVEL).reshape(self.n, 6, -1)[:, 0]
        self.flat = np.hstack([race, f32(df, self.per_boat).reshape(self.n, -1)])
        self.flat_names = feat.RACE_LEVEL + [f"{c}_b{b}" for b in range(1, 7) for c in self.per_boat]
        self.dates = df["date"].to_numpy().reshape(-1, 6)[:, 0]

    # ---- 1着段（艇単位）: 自艇 + 隣 + 全艇の主要特徴 ----
    def first_rows(self, idx):
        x = np.hstack([self.boat[idx].reshape(len(idx) * 6, -1), np.repeat(self.lineup[idx], 6, axis=0)])
        return x, self.boat_names + self.lineup_names

    # ---- 2着段（艇単位）: 1着艇を所与とした残り5艇 ----
    def second_rows(self, idx, winner):
        cand = np.array([[b for b in range(6) if b != w] for w in winner])
        r, c, w = np.repeat(idx, 5), cand.ravel(), np.repeat(winner, 5)
        x = np.hstack([self.boat[r, c], self.main[r, w], self.main[r, c] - self.main[r, w], self.lineup[r],
                       w[:, None].astype(np.float32), (c - w)[:, None].astype(np.float32),
                       (c < w)[:, None].astype(np.float32)])
        names = (self.boat_names + [f"{c}_W" for c in MAIN] + [f"{c}_dW" for c in MAIN] + self.lineup_names
                 + ["winner", "lane_gap", "inside_winner"])
        return x, names, cand


CATEGORICAL = {"cl1": ["jcd", "boat"], "flat1": ["jcd"], "cl2": ["jcd", "boat", "winner"]}


class RaceModelV3:
    def _data(self, kind, d, idx, order):
        """kind ごとの (X, y, 列名, パラメータ)。order は (レース, 6) の着順どおりの艇番号。"""
        first, second = order[idx, 0], order[idx, 1]
        if kind == "cl1":
            x, names = d.first_rows(idx)
            y = (np.arange(6)[None] == first[:, None]).astype(np.float64).ravel()
            return x, y, names, dict(PARAMS_CL, objective=objective(6)), metric(6)
        if kind == "flat1":
            return d.flat[idx], first, d.flat_names, PARAMS_FLAT, None
        x, names, cand = d.second_rows(idx, first)
        y = (cand == second[:, None]).astype(np.float64).ravel()
        return x, y, names, dict(PARAMS_CL, objective=objective(5)), metric(5)

    def _train(self, kind, d, idx, order, rounds, valid=None):
        x, y, names, params, feval = self._data(kind, d, idx, order)
        train = lgb.Dataset(x, y, feature_name=names, categorical_feature=CATEGORICAL[kind], free_raw_data=True)
        if valid is None:
            return lgb.train(params, train, rounds, feval=feval)
        xv, yv, *_ = self._data(kind, d, valid, order)
        return lgb.train(params, train, rounds, valid_sets=[lgb.Dataset(xv, yv, reference=train)], feval=feval,
                         callbacks=[lgb.early_stopping(100, verbose=False)])

    def find_iterations(self, d, order):
        """学習末尾 EARLY_STOP_DAYS 日を検証に回して早期停止した本数。"""
        last = datetime.strptime(d.dates.max(), "%Y%m%d")
        cut = (last - timedelta(days=EARLY_STOP_DAYS)).strftime("%Y%m%d")
        idx = np.arange(d.n)
        fit, valid = idx[d.dates <= cut], idx[d.dates > cut]
        return {kind: int(self._train(kind, d, fit, order, 5000, valid).best_iteration)
                for kind in ("cl1", "flat1", "cl2")}

    def fit(self, df, iterations=None, log=print):
        d = Design(df)
        order = np.argsort(df["rank"].to_numpy().reshape(-1, 6), axis=1)
        self.per_boat = d.per_boat
        self.early_stop_iterations = iterations or self.find_iterations(d, order)
        log(f"  early-stopped iterations: {self.early_stop_iterations}")
        idx = np.arange(d.n)
        self.models = {}
        for kind, it in self.early_stop_iterations.items():
            booster = self._train(kind, d, idx, order, int(round(it * REFIT_FACTOR)))
            self.models[kind] = booster.model_to_string()
        self.training_through = str(d.dates.max())
        self.train_races = d.n
        self._boosters = None
        return self

    def boosters(self):
        if getattr(self, "_boosters", None) is None:
            self._boosters = {k: lgb.Booster(model_str=s) for k, s in self.models.items()}
        return self._boosters

    def predict(self, df):
        d = Design(df, self.per_boat)
        b = self.boosters()
        idx = np.arange(d.n)
        x, _ = d.first_rows(idx)
        p_cl = group_softmax(b["cl1"].predict(x, num_threads=THREADS), 6)
        p_flat = b["flat1"].predict(d.flat, num_threads=THREADS)
        logp = BLEND_CL * np.log(np.clip(p_cl, 1e-12, 1)) + (1 - BLEND_CL) * np.log(np.clip(p_flat, 1e-12, 1))
        p1 = np.exp(logp - logp.max(axis=1, keepdims=True))
        p1 /= p1.sum(axis=1, keepdims=True)
        cond = np.zeros((d.n, 6, 6))
        for w in range(6):
            x, _, cand = d.second_rows(idx, np.full(d.n, w))
            cond[idx[:, None], w, cand] = group_softmax(b["cl2"].predict(x, num_threads=THREADS), 5)
        joint = p1[:, PAIRS[:, 0]] * cond[:, PAIRS[:, 0], PAIRS[:, 1]]
        if not np.isfinite(joint).all() or not np.allclose(joint.sum(axis=1), 1):
            raise ValueError("Invalid probabilities")
        return p1, joint


def save_model(model, path):
    payload = pickle.dumps(dict(version="live_v3", models=model.models, per_boat=model.per_boat,
                                early_stop_iterations=model.early_stop_iterations,
                                training_through=model.training_through, train_races=model.train_races))
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path):
    # ローカルで生成したものだけを読む。pickle は実行可能データ
    with path.open("rb") as f:
        saved = pickle.load(f)
    if saved.get("version") != "live_v3":
        raise ValueError("Not a live_v3 artifact")
    model = RaceModelV3()
    model.models, model.per_boat = saved["models"], saved["per_boat"]
    model.early_stop_iterations = saved["early_stop_iterations"]
    model.training_through, model.train_races = saved["training_through"], saved["train_races"]
    return model


def final_market_loss(keys, order):
    """確定市場の log-loss（的中目の払戻から）。通常払戻が1件のレースだけ。列は 2連単・2連複。"""
    import live_market_v2 as m2
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet")
    out = np.full((keys.height, 2), np.nan)
    for j, pool in enumerate(("2t", "2f")):
        pay = (payouts.filter(pl.col("bet_type") == m2.LABELS[pool]).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        pay = keys.join(pay, on=KEY, how="left", maintain_order="left")
        a, b = order[:, 0] + 1, order[:, 1] + 1
        win = [f"{x}-{y}" if pool == "2t" else f"{min(x, y)}-{max(x, y)}" for x, y in zip(a, b)]
        ok = (pay["n"].to_numpy() == 1) & (pay["combo"].to_numpy() == np.array(win))
        amount = pay["amount"].to_numpy().astype(float)
        out[ok, j] = -np.log((100 / amount[ok]) / m2.TAKE[pool])
    return out


def compare(frame, p1, joint, baseline_losses=None):
    """log-loss（1着・2連単・2連複）と、確定市場・v2 との差。"""
    losses = v1.losses(frame, p1, joint)
    keys = frame.select(KEY).unique(maintain_order=True)
    dates = keys["date"].to_numpy()
    order = np.argsort(frame["rank"].to_numpy().reshape(-1, 6), axis=1)
    market = final_market_loss(keys, order)
    report = dict(races=keys.height, period=[keys["date"].min(), keys["date"].max()],
                  order=["1着", "2連単", "2連複"], logloss=losses.mean(axis=0).tolist())
    for j, pool in enumerate(("2連単", "2連複"), start=1):
        ok = np.isfinite(market[:, j - 1])
        diff = losses[ok, j] - market[ok, j - 1]
        report[f"minus_final_market_{pool}"] = dict(
            races=int(ok.sum()), final_market_logloss=float(market[ok, j - 1].mean()), mean=float(diff.mean()),
            day_bootstrap_95ci=v2.day_bootstrap(diff[:, None], dates[ok])[0])
    if baseline_losses is not None:
        diff = losses - baseline_losses
        report["v2_logloss"] = baseline_losses.mean(axis=0).tolist()
        report["minus_v2"] = dict(mean=diff.mean(axis=0).tolist(), day_bootstrap_95ci=v2.day_bootstrap(diff, dates))
    return report, losses


V2_DATA_START = "20220701"  # v2 を作った時点のデータ先頭。v2 の履歴特徴はここから数える


def v2_frame(end, entries=None, results=None):
    """live_v2.build_frame と同じ。ただし v2 が見ていたデータ範囲（20220701〜）に絞る。

    v3 のために 2019-07 まで遡ったので、絞らないと v2 の履歴特徴（通算走数など）が学習時と違う分布になる。
    live_v2.py は凍結対象なので書き換えず、ここに写す。
    """
    from build_features import racer_history, venue_boat_history
    span = (pl.col("date") >= V2_DATA_START) & (pl.col("date") <= end)
    entries = (pl.read_parquet(ROOT / "data/parsed/entries.parquet") if entries is None else entries).filter(
        span).with_columns(class_ord=pl.col("class").replace_strict(v2.baseline.CLASS_ORD, default=None))
    results = (pl.read_parquet(ROOT / "data/parsed/results.parquet") if results is None else results).filter(span)
    historic = results.with_columns(st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64))
    rh, vb = racer_history(historic), venue_boat_history(historic)
    labels = results.select(*v2.BK, "racer_id", "tenji",
                            rank=pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8))
    df = (entries.join(labels, on=v2.BK + ["racer_id"], how="inner", validate="1:1")
          .join(rh, on=["racer_id", "date"], how="left", validate="m:1")
          .join(vb, on=["jcd", "boat", "date"], how="left", validate="m:1"))
    normal = (df.group_by(KEY)
              .agg(n=pl.len(), ranked=pl.col("rank").is_not_null().sum(), distinct=pl.col("rank").n_unique())
              .filter((pl.col("n") == 6) & (pl.col("ranked") == 6) & (pl.col("distinct") == 6)).select(KEY))
    return v2.derive(df.join(normal, on=KEY).filter(pl.col("date") >= v2.TRAIN_START).sort(v2.BK))


def dev():
    """封印前だけの比較。学習 〜20240630 / 検証 20240701〜20250718。v2 も同じ区間で学習し直す。"""
    if DEV["valid_through"] >= HOLDOUT_START:
        raise SystemExit("Development window overlaps the sealed holdout")
    df = load_frame(hi=DEV["valid_through"])
    train, valid = df.filter(pl.col("date") < DEV["train_before"]), df.filter(pl.col("date") >= DEV["train_before"])
    model = RaceModelV3().fit(train)
    p1, joint = model.predict(valid)
    # v2: 同じ検証レースで（v2 の学習は 20220901〜。v2 の特徴量・モデルは変えない）
    f2 = v2_frame(DEV["valid_through"])
    v2_model = v2.RaceModelV2().fit(f2.filter(pl.col("date") < DEV["train_before"]))
    f2_valid = f2.filter(pl.col("date") >= DEV["train_before"])
    if not f2_valid.select(KEY).unique(maintain_order=True).equals(valid.select(KEY).unique(maintain_order=True)):
        raise ValueError("v2 and v3 development races differ")
    report, _ = compare(valid, p1, joint, v1.losses(f2_valid, *v2_model.predict(f2_valid)))
    report.update(train_races=train.height // 6, train_period=[train["date"].min(), train["date"].max()],
                  iterations=model.early_stop_iterations, status="development (walk-forward side only)",
                  recorded_at=datetime.now(timezone.utc).isoformat())
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "prereg/live_v3_dev.json").write_text(text)
    print(text)


def train():
    df = load_frame()
    report = dict(training_start=TRAIN_START, training_end_inclusive=END,
                  sealed_period_used_for="training only (never evaluated)",
                  blend_cl=BLEND_CL, refit_factor=REFIT_FACTOR, early_stop_days=EARLY_STOP_DAYS, folds=[])
    model = RaceModelV3().fit(df)
    report["train_races"], report["iterations"] = model.train_races, model.early_stop_iterations
    # 合成器の推定用に、時系列外の予測を作る。本数は最終モデルのものを使い回す（学習量がほぼ同じ）
    folds = []
    for train_end, lo, hi in OOS_FOLDS:
        tr = df.filter(pl.col("date") < train_end)
        te = df.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        p1, joint = RaceModelV3().fit(tr, model.early_stop_iterations).predict(te)
        keys = te.select(KEY).unique(maintain_order=True)
        folds.append(keys.with_columns(
            p1=pl.Series(p1.tolist()), joint=pl.Series(joint.tolist()),
            rank=pl.Series(te["rank"].to_numpy().reshape(-1, 6).tolist()), model_trained_before=pl.lit(train_end)))
        fold_report, _ = compare(te, p1, joint)
        report["folds"].append(dict(trained_before=train_end, train_races=tr.height // 6, **fold_report))
        print(json.dumps(report["folds"][-1], ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(folds).write_parquet(OUT / "oos_predictions.parquet")
    report["model_sha256"] = save_model(model, OUT / "model.pkl")
    report["oos_sha256"] = hashlib.sha256((OUT / "oos_predictions.parquet").read_bytes()).hexdigest()
    report["versions"] = {"lightgbm": lgb.__version__, "polars": pl.__version__, "numpy": np.__version__}
    report["recorded_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (OUT / "metadata.json").write_text(text)
    (ROOT / "prereg/live_v3_train.json").write_text(text)
    print(f"Saved model trained on {model.train_races} races through {model.training_through}")


def live_frame(inputs, entries=None, results=None, meta=None):
    """スクショ転記（展示タイム）＋ 当日の公式 B（出走表）＋ 前日までの結果 → 特徴量。"""
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet") if entries is None else entries
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet") if results is None else results
    meta = pl.read_parquet(ROOT / "data/parsed/race_meta.parquet") if meta is None else meta
    keys = inputs.select(KEY).unique()
    official = entries.join(keys, on=KEY)
    if official.height != keys.height * 6:
        raise ValueError("当日の公式 B（出走表）が entries.parquet に無い。download_lzh → extract_lzh → "
                         "import_bk_range → parse_b_meta の順に取り込む")
    check = inputs.select(*BK, "racer_id").join(official.select(*BK, official_id="racer_id"), on=BK)
    if check.height != inputs.height or not (check["racer_id"] == check["official_id"]).all():
        raise ValueError("Racer ids in the input do not match the official entries")
    if meta.join(keys, on=KEY).height != keys.height:
        raise ValueError("race_meta.parquet に対象レースが無い。parse_b_meta.py を流す")
    dates = inputs["date"].unique().to_list()
    # 対象日以降の結果は使わない（あっても集計に入らないが、明示的に落とす）
    frame = feat.build(entries.filter(pl.col("date") <= max(dates)), results.filter(pl.col("date") < min(dates)),
                       meta, tenji=inputs.select(*BK, "tenji"))
    return frame.join(keys, on=KEY).sort(BK)


def predict(path, out, model_path=OUT / "model.pkl"):
    inputs = v2.validate_input(pl.read_csv(path, schema_overrides={"date": pl.String}))
    model = load_model(model_path)
    if inputs["date"].min() <= model.training_through:
        raise ValueError("Prediction date must be after model training cutoff")
    if out.exists():
        raise ValueError("Refusing to overwrite prediction")
    target = live_frame(inputs)
    p1, joint = model.predict(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    v2.probability_rows(target.select(KEY).unique(maintain_order=True), p1, joint).write_csv(out)
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    out.with_suffix(".json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v3",
        training_through=model.training_through,
        history_through=results.filter(pl.col("date") < inputs["date"].min())["date"].max(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        note="Odds not used. Entries come from the official B file; history uses results strictly before the "
             "race date. history_through shows how stale it is."), indent=2) + "\n")
    return p1, joint


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["dev", "train", "predict"])
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "dev":
        dev()
    elif args.command == "train":
        train()
    elif args.input and args.out:
        predict(args.input, args.out)
    else:
        ap.error("predict requires --input and --out")
