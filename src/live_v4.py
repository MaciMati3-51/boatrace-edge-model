"""直前情報モデル v4 — v3 にレーティング特徴・3着段・シード平均を足す。

v3（live_v3.py・v3_features.py）は前向き検証の凍結対象なので書き換えない。特徴量キャッシュ
（data/features/v3.parquet）と関数は借りる。

v3 からの変更（数値は開発区間: 学習 〜20240630、検証 20240701〜20250718。封印期間は見ていない。
詳細と探索の記録は prereg/live_v4_design.md）:

1. レーティング特徴（v4_ratings.py）: 相手の強さ・枠・モーターを同時に引いた選手の実力と、選手の腕を引いた
   モーターの力。逐次推定の Plackett-Luce。v3 の成績は「会場×枠の平均との差」までしか引いていなかった
2. 3着段: 1・2着艇を所与とした残り4艇の softmax。3連単120通り・3連複20通りの確率が出る
3. シード平均: 艇単位の木を SEEDS 本ずつ育て、スコアを平均する

特徴量: .venv/bin/python src/v3_features.py           （entries/results を更新したら先に流す。v3 と共通）
        .venv/bin/python src/live_v4.py features      （履歴を 2019-07 より前へ延ばしている場合はこちらも）
開発  : .venv/bin/python src/live_v4.py dev            （封印前だけで v3 と比較）
学習  : .venv/bin/python src/live_v4.py train [--end YYYYMMDD] [--tag NAME] [--fold 学習打ち切り 開始 終了]
時系列外: .venv/bin/python src/live_v4.py oos --tag NAME --lo YYYYMMDD --hi YYYYMMDD
予測  : .venv/bin/python src/live_v4.py predict --input race.csv --out pred.csv
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from holdout import HOLDOUT_START
import live_model as v1
import live_v2 as v2
import live_v3 as v3
import v3_features as feat
import v4_meta
import v4_ratings as rt

ROOT = Path(__file__).resolve().parents[1]
KEY, BK = feat.KEY, feat.BK
OUT = ROOT / "data/models/live_v4"
PAIRS = v3.PAIRS                                                    # 2連単30通り（0始まり）
TRIPLES = np.array(list(itertools.permutations(range(6), 3)))       # 3連単120通り（0始まり）
TRIOS = np.array(list(itertools.combinations(range(6), 3)))         # 3連複20通り
CAND3 = np.array([[[b for b in range(6) if b not in (w, s)][:4] for s in range(6)] for w in range(6)])
# 履歴の先頭。20190701 なら v3 と同じデータ（特徴量キャッシュも v3 のものを使う）。それより前にすると
# data/parsed/ext/（build_history_ext.py）を足し、特徴量は data/features/v4.parquet に作り直す
HISTORY_START, MAIN_START = "20190701", "20190701"
TRAIN_START, END = v3.TRAIN_START, "20261112"
FEATURE_CACHE = ROOT / ("data/features/v3.parquet" if HISTORY_START == MAIN_START else "data/features/v4.parquet")
DEV = v3.DEV
EARLY_STOP_DAYS, REFIT_FACTOR = v3.EARLY_STOP_DAYS, v3.REFIT_FACTOR
BLEND_CL = 0.7   # 1着段: 艇単位 / 6クラス の幾何平均の重み
SEEDS = (0, 1, 2)
THREADS = 8
TAKE = {"2t": 1.3407, "2f": 1.345, "3t": 1.3364, "3f": 1.345}  # Σ(1/オッズ)。3連複は板が無いので2連複の値を仮置き

RATING_REL = ["rt_s_theta", "rt_f_theta", "rt_s_racer", "rt_s_motor"]
RATING_MAIN = ["rt_s_theta", "rt_f_theta"]
REL = v3.REL + RATING_REL
MAIN = v3.MAIN + RATING_MAIN
META4 = ["fixed4", "rtype4"]              # v4_meta.py: 進入固定の旗と、表記ゆれを直したレース種別
RACE_LEVEL = feat.RACE_LEVEL + META4
N_MAX = 10  # MAIN の先頭 N_MAX 列はレース内の最大値も持たせる（v3 と同じ）
PARAMS_CL, PARAMS_FLAT = v3.PARAMS_CL, v3.PARAMS_FLAT
KINDS = ("cl1", "flat1", "cl2", "cl3")
GROUP = {"cl1": 6, "cl2": 5, "cl3": 4}
CATEGORICAL = {"cl1": ["jcd", "boat"], "flat1": ["jcd"], "cl2": ["jcd", "boat", "winner"],
               "cl3": ["jcd", "boat", "winner", "second"]}
group_softmax, objective, metric = v3.group_softmax, v3.objective, v3.metric


def history():
    """番組表・結果・レース情報。HISTORY_START が本体より前なら data/parsed/ext/ を足す。"""
    tabs = {}
    for name in ("entries", "results", "race_meta"):
        df = pl.read_parquet(ROOT / f"data/parsed/{name}.parquet")
        if HISTORY_START < MAIN_START:
            old = pl.read_parquet(ROOT / f"data/parsed/ext/{name}.parquet").filter(pl.col("date") >= HISTORY_START)
            df = pl.concat([old, df.filter(pl.col("date") >= MAIN_START)])
        tabs[name] = df
    return tabs["entries"], tabs["results"], tabs["race_meta"].unique(subset=KEY, keep="last")


def build_features():
    """v3 と同じ特徴量を、延長した履歴で作り直す（HISTORY_START が本体より前のときだけ要る）。"""
    if HISTORY_START == MAIN_START:
        raise SystemExit("HISTORY_START equals the main start; v4 reuses data/features/v3.parquet")
    entries, results, meta = history()
    df = feat.build(entries, results, meta).join(feat.labels(results), on=BK + ["racer_id"], how="left")
    df.write_parquet(FEATURE_CACHE)
    print(f"{FEATURE_CACHE}: {df.height // 6:,} races, {df['date'].min()}〜{df['date'].max()}")


def add_ratings(df, entries=None, results=None, meta4=None):
    """特徴量フレームに v4 の追加列を付ける: レーティング（対象日より前の結果だけで決まる）と v4_meta の2列。"""
    hi = df["date"].max()
    if entries is None or results is None:
        entries, results, _ = history()
    ratings = rt.build(entries.filter(pl.col("date") <= hi), results.filter(pl.col("date") <= hi))
    meta4 = pl.read_parquet(v4_meta.OUT) if meta4 is None else meta4
    out = (df.join(ratings, on=BK, how="left").join(meta4.select(*KEY, *META4), on=KEY, how="left")).sort(BK)
    if out["rt_s_theta"].null_count():
        raise ValueError("Ratings missing for some rows")
    if out["fixed4"].null_count():
        raise ValueError("race_meta_v4.parquet に対象レースが無い。v4_meta.py を流す")
    return out


def load_frame(lo=TRAIN_START, hi=END):
    """正常レース（6艇完走）の特徴量＋レーティング。1艇1行、レース内は枠順。"""
    df = pl.read_parquet(FEATURE_CACHE).filter(
        pl.col("rank").is_not_null() & (pl.col("date") >= lo) & (pl.col("date") <= hi))
    # 出走表と結果で選手が違う艇（欠場の差し替え）があるレースは6行揃わないので落とす（v3 と同じ）
    full = df.group_by(KEY).len().filter(pl.col("len") == 6).select(KEY)
    return add_ratings(df.join(full, on=KEY).sort(BK))


class Design:
    """特徴量フレーム → 学習・予測用の配列（v3 の Design にレーティング列と3着段を足したもの）。"""

    def __init__(self, df, per_boat=None):
        df = df.with_columns(
            [(pl.col(c) - pl.col(c).mean().over(KEY)).alias(f"{c}_rel") for c in REL]
            + [pl.col(c).rank("average").over(KEY).alias(f"{c}_rk") for c in REL])
        self.per_boat = per_boat or [c for c in feat.per_boat_columns(df) if c not in META4]
        missing = [c for c in self.per_boat + RACE_LEVEL if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing: {missing}")
        near = df
        for shift, tag in ((1, "in"), (-1, "out")):  # 1つ内側・1つ外側の艇
            near = near.with_columns([pl.col(c).shift(shift).over(KEY).alias(f"{c}_{tag}") for c in MAIN])
        near = near.with_columns([pl.col(c).max().over(KEY).alias(f"{c}_max") for c in MAIN[:N_MAX]])
        extra = [f"{c}_{t}" for c in MAIN for t in ("in", "out")] + [f"{c}_max" for c in MAIN[:N_MAX]]
        self.boat_names = ["boat"] + RACE_LEVEL + self.per_boat + extra
        self.n = df.height // 6
        f32 = lambda frame, cols: frame.select(cols).to_numpy().astype(np.float32)
        self.boat = f32(near, self.boat_names).reshape(self.n, 6, -1)
        self.main = f32(df, MAIN).reshape(self.n, 6, -1)
        self.lineup = self.main.reshape(self.n, -1)
        self.lineup_names = [f"{c}_L{b}" for b in range(1, 7) for c in MAIN]
        race = f32(df, RACE_LEVEL).reshape(self.n, 6, -1)[:, 0]
        self.flat = np.hstack([race, f32(df, self.per_boat).reshape(self.n, -1)])
        self.flat_names = RACE_LEVEL + [f"{c}_b{b}" for b in range(1, 7) for c in self.per_boat]
        self.dates = df["date"].to_numpy().reshape(-1, 6)[:, 0]

    # ---- 1着段: 自艇 + 隣 + 全艇の主要特徴 ----
    def first_rows(self, idx):
        x = np.hstack([self.boat[idx].reshape(len(idx) * 6, -1), np.repeat(self.lineup[idx], 6, axis=0)])
        return x, self.boat_names + self.lineup_names

    # ---- 2着段: 1着艇を所与とした残り5艇 ----
    def second_rows(self, idx, winner):
        cand = np.array([[b for b in range(6) if b != w] for w in range(6)])[winner]
        r, c, w = np.repeat(idx, 5), cand.ravel(), np.repeat(winner, 5)
        col = lambda v: v[:, None].astype(np.float32)
        x = np.hstack([self.boat[r, c], self.main[r, w], self.main[r, c] - self.main[r, w], self.lineup[r],
                       col(w), col(c - w), col(c < w)])
        names = (self.boat_names + [f"{c}_W" for c in MAIN] + [f"{c}_dW" for c in MAIN] + self.lineup_names
                 + ["winner", "lane_gap", "inside_winner"])
        return x, names, cand

    # ---- 3着段: 1・2着艇を所与とした残り4艇 ----
    def third_rows(self, idx, winner, second):
        cand = CAND3[winner, second]
        r, c, w, s = np.repeat(idx, 4), cand.ravel(), np.repeat(winner, 4), np.repeat(second, 4)
        col = lambda v: v[:, None].astype(np.float32)
        x = np.hstack([self.boat[r, c], self.main[r, w], self.main[r, s], self.main[r, c] - self.main[r, w],
                       self.main[r, c] - self.main[r, s], self.lineup[r], col(w), col(s), col(c - w), col(c - s),
                       col(c < w), col(c < s)])
        names = (self.boat_names + [f"{m}_W" for m in MAIN] + [f"{m}_S" for m in MAIN]
                 + [f"{m}_dW" for m in MAIN] + [f"{m}_dS" for m in MAIN] + self.lineup_names
                 + ["winner", "second", "gap_w", "gap_s", "inside_w", "inside_s"])
        return x, names, cand


class RaceModelV4:
    def _data(self, kind, d, idx, order):
        """kind ごとの (X, y, 列名)。order は (レース, 6) の着順どおりの艇番号。"""
        first, second, third = order[idx, 0], order[idx, 1], order[idx, 2]
        if kind == "flat1":
            return d.flat[idx], first, d.flat_names
        if kind == "cl1":
            x, names = d.first_rows(idx)
            return x, (np.arange(6)[None] == first[:, None]).astype(np.float64).ravel(), names
        if kind == "cl2":
            x, names, cand = d.second_rows(idx, first)
            return x, (cand == second[:, None]).astype(np.float64).ravel(), names
        x, names, cand = d.third_rows(idx, first, second)
        return x, (cand == third[:, None]).astype(np.float64).ravel(), names

    @staticmethod
    def _params(kind, seed):
        if kind == "flat1":
            return dict(PARAMS_FLAT, seed=seed), None
        return dict(PARAMS_CL, seed=seed, objective=objective(GROUP[kind])), metric(GROUP[kind])

    def _train(self, kind, d, idx, order, rounds, valid=None, seed=0):
        x, y, names = self._data(kind, d, idx, order)
        params, feval = self._params(kind, seed)
        train = lgb.Dataset(x, y, feature_name=names, categorical_feature=CATEGORICAL[kind], free_raw_data=False)
        if valid is not None:
            xv, yv, _ = self._data(kind, d, valid, order)
            return [lgb.train(params, train, rounds, valid_sets=[lgb.Dataset(xv, yv, reference=train)],
                              feval=feval, callbacks=[lgb.early_stopping(100, verbose=False)])]
        # 同じ Dataset を使い回してシードだけ替える（6クラス側は1本）
        seeds = (seed,) if kind == "flat1" else SEEDS
        return [lgb.train(self._params(kind, s)[0], train, rounds, feval=feval) for s in seeds]

    def find_iterations(self, d, order):
        """学習末尾 EARLY_STOP_DAYS 日を検証に回して早期停止した本数。"""
        last = datetime.strptime(d.dates.max(), "%Y%m%d")
        cut = (last - timedelta(days=EARLY_STOP_DAYS)).strftime("%Y%m%d")
        idx = np.arange(d.n)
        fit, valid = idx[d.dates <= cut], idx[d.dates > cut]
        return {kind: int(self._train(kind, d, fit, order, 5000, valid)[0].best_iteration) for kind in KINDS}

    def fit(self, df, iterations=None, log=print):
        d = Design(df)
        order = np.argsort(df["rank"].to_numpy().reshape(-1, 6), axis=1)
        self.per_boat = d.per_boat
        self.early_stop_iterations = iterations or self.find_iterations(d, order)
        log(f"  early-stopped iterations: {self.early_stop_iterations}")
        idx = np.arange(d.n)
        self.models = {}
        for kind, it in self.early_stop_iterations.items():
            boosters = self._train(kind, d, idx, order, int(round(it * REFIT_FACTOR)))
            self.models[kind] = [b.model_to_string() for b in boosters]
            log(f"  trained {kind} x{len(boosters)}")
        self.training_through = str(d.dates.max())
        self.train_races = d.n
        self._boosters = None
        return self

    def boosters(self):
        if getattr(self, "_boosters", None) is None:
            self._boosters = {k: [lgb.Booster(model_str=s) for s in v] for k, v in self.models.items()}
        return self._boosters

    def _score(self, kind, x):
        return np.mean([b.predict(x, num_threads=THREADS) for b in self.boosters()[kind]], axis=0)

    def predict(self, df, third=True):
        """→ (1着 (n,6), 2連単 (n,30) PAIRS 順, 3連単 (n,120) TRIPLES 順。third=False なら None)"""
        d = Design(df, self.per_boat)
        idx = np.arange(d.n)
        x, _ = d.first_rows(idx)
        p_cl = group_softmax(self._score("cl1", x), 6)
        p_flat = self._score("flat1", d.flat)
        logp = BLEND_CL * np.log(np.clip(p_cl, 1e-12, 1)) + (1 - BLEND_CL) * np.log(np.clip(p_flat, 1e-12, 1))
        p1 = np.exp(logp - logp.max(axis=1, keepdims=True))
        p1 /= p1.sum(axis=1, keepdims=True)
        cond2 = np.zeros((d.n, 6, 6))
        for w in range(6):
            x, _, cand = d.second_rows(idx, np.full(d.n, w))
            cond2[idx[:, None], w, cand] = group_softmax(self._score("cl2", x), 5)
        joint = p1[:, PAIRS[:, 0]] * cond2[:, PAIRS[:, 0], PAIRS[:, 1]]
        if not np.isfinite(joint).all() or not np.allclose(joint.sum(axis=1), 1):
            raise ValueError("Invalid probabilities")
        if not third:
            return p1, joint, None
        cond3 = np.zeros((d.n, 6, 6, 6))
        for w, s in PAIRS:
            x, _, cand = d.third_rows(idx, np.full(d.n, w), np.full(d.n, s))
            cond3[idx[:, None], w, s, cand] = group_softmax(self._score("cl3", x), 4)
        a, b, c = TRIPLES.T
        tri = p1[:, a] * cond2[:, a, b] * cond3[:, a, b, c]
        if not np.isfinite(tri).all() or not np.allclose(tri.sum(axis=1), 1):
            raise ValueError("Invalid trifecta probabilities")
        return p1, joint, tri


def trio_from_trifecta(tri):
    """3連単120通り（TRIPLES 順）→ 3連複20通り（TRIOS 順）。"""
    key = {tuple(t): i for i, t in enumerate(map(tuple, TRIOS))}
    col = np.array([key[tuple(sorted(t))] for t in TRIPLES])
    out = np.zeros(tri.shape[:-1] + (20,))
    for j in range(20):
        out[..., j] = tri[..., col == j].sum(axis=-1)
    return out


def save_model(model, path):
    payload = pickle.dumps(dict(version="live_v4", models=model.models, per_boat=model.per_boat,
                                early_stop_iterations=model.early_stop_iterations,
                                training_through=model.training_through, train_races=model.train_races))
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path):
    # ローカルで生成したものだけを読む。pickle は実行可能データ
    with path.open("rb") as f:
        saved = pickle.load(f)
    if saved.get("version") != "live_v4":
        raise ValueError("Not a live_v4 artifact")
    model = RaceModelV4()
    model.models, model.per_boat = saved["models"], saved["per_boat"]
    model.early_stop_iterations = saved["early_stop_iterations"]
    model.training_through, model.train_races = saved["training_through"], saved["train_races"]
    return model


# ---------------------------------------------------------------- 評価
NAMES = ["1着", "2連単", "2連複", "3連単", "3連複"]
BET = {"2連単": ("２連単", "2t"), "2連複": ("２連複", "2f"), "3連単": ("３連単", "3t"), "3連複": ("３連複", "3f")}


def losses(order, p1, joint, tri=None):
    """レースごとの log-loss。列は NAMES の順（tri が無ければ先頭3列）。"""
    r = np.arange(len(order))
    pair = {tuple(p): i for i, p in enumerate(map(tuple, PAIRS))}
    j = np.array([pair[(a, b)] for a, b in order[:, :2]])
    jr = np.array([pair[(b, a)] for a, b in order[:, :2]])
    clip = lambda p: -np.log(np.clip(p, 1e-15, 1))
    cols = [clip(p1[r, order[:, 0]]), clip(joint[r, j]), clip(joint[r, j] + joint[r, jr])]
    if tri is not None:
        triple = {tuple(t): i for i, t in enumerate(map(tuple, TRIPLES))}
        trio = {tuple(t): i for i, t in enumerate(map(tuple, TRIOS))}
        k = np.array([triple[tuple(o)] for o in order[:, :3]])
        kf = np.array([trio[tuple(sorted(o))] for o in order[:, :3]])
        cols += [clip(tri[r, k]), clip(trio_from_trifecta(tri)[r, kf])]
    return np.column_stack(cols)


def final_market_losses(keys, order, payouts=None):
    """確定市場の log-loss（的中目の払戻から）。通常払戻が1件のレースだけ、他は nan。列は BET の順。"""
    payouts = pl.read_parquet(ROOT / "data/parsed/payouts.parquet") if payouts is None else payouts
    out = np.full((keys.height, len(BET)), np.nan)
    o = order[:, :3] + 1
    for j, (label, pool) in enumerate(BET.values()):
        pay = (payouts.filter(pl.col("bet_type") == label).group_by(KEY)
               .agg(n=pl.len(), combo=pl.col("combo").first(), amount=pl.col("amount").first()))
        pay = keys.join(pay, on=KEY, how="left", maintain_order="left")
        if pool == "2t":
            win = [f"{a}-{b}" for a, b, _ in o]
        elif pool == "2f":
            win = [f"{min(a, b)}-{max(a, b)}" for a, b, _ in o]
        elif pool == "3t":
            win = [f"{a}-{b}-{c}" for a, b, c in o]
        else:
            win = ["-".join(map(str, sorted(t))) for t in o]
        amount = pay["amount"].to_numpy().astype(float)
        ok = (pay["n"].to_numpy() == 1) & (pay["combo"].to_numpy() == np.array(win)) & (np.nan_to_num(amount) > 0)
        out[ok, j] = -np.log((100 / amount[ok]) / TAKE[pool])
    return out


def compare(frame, p1, joint, tri, baseline_losses=None, baseline="v3"):
    """log-loss と、確定市場・基準モデルとの差（日単位ブートストラップ）。"""
    order = np.argsort(frame["rank"].to_numpy().reshape(-1, 6), axis=1)
    ls = losses(order, p1, joint, tri)
    keys = frame.select(KEY).unique(maintain_order=True)
    dates = keys["date"].to_numpy()
    market = final_market_losses(keys, order)
    names = NAMES[:ls.shape[1]]
    report = dict(races=keys.height, period=[keys["date"].min(), keys["date"].max()], order=names,
                  logloss=ls.mean(axis=0).tolist(), minus_final_market={})
    for j, pool in enumerate(BET):
        if j + 1 >= ls.shape[1]:
            break
        ok = np.isfinite(market[:, j])
        diff = ls[ok, j + 1] - market[ok, j]
        report["minus_final_market"][pool] = dict(
            races=int(ok.sum()), final_market_logloss=float(market[ok, j].mean()), mean=float(diff.mean()),
            day_bootstrap_95ci=v2.day_bootstrap(diff[:, None], dates[ok])[0])
    if baseline_losses is not None:
        diff = ls[:, :3] - baseline_losses[:, :3]
        report[f"{baseline}_logloss"] = baseline_losses.mean(axis=0).tolist()
        report[f"minus_{baseline}"] = dict(order=NAMES[:3], mean=diff.mean(axis=0).tolist(),
                                           day_bootstrap_95ci=v2.day_bootstrap(diff, dates))
    return report, ls


def dev():
    """封印前だけの比較。学習 〜20240630 / 検証 20240701〜20250718。v3 も同じ区間で学習し直す。"""
    if DEV["valid_through"] >= HOLDOUT_START:
        raise SystemExit("Development window overlaps the sealed holdout")
    df = load_frame(hi=DEV["valid_through"])
    train, valid = df.filter(pl.col("date") < DEV["train_before"]), df.filter(pl.col("date") >= DEV["train_before"])
    cache = ROOT / "data/preds/live_v3_dev_losses.npy"  # v3 の検証区間のレース別 log-loss（学習し直しは1回だけ）
    if cache.exists():
        base = np.load(cache)
    else:
        f3 = df.drop(rt.COLUMNS)
        m3 = v3.RaceModelV3().fit(f3.filter(pl.col("date") < DEV["train_before"]))
        base = v1.losses(f3.filter(pl.col("date") >= DEV["train_before"]),
                         *m3.predict(f3.filter(pl.col("date") >= DEV["train_before"])))
        np.save(cache, base)
    model = RaceModelV4().fit(train)
    report, _ = compare(valid, *model.predict(valid), baseline_losses=base)
    report.update(train_races=train.height // 6, train_period=[train["date"].min(), train["date"].max()],
                  iterations=model.early_stop_iterations, seeds=list(SEEDS), blend_cl=BLEND_CL,
                  status="development (walk-forward side only)", recorded_at=datetime.now(timezone.utc).isoformat())
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "prereg/live_v4_dev.json").write_text(text)
    print(text)


def train(end=END, tag="live_v4", folds=()):
    """end までの全データで学習して data/models/<tag>/ に保存する。

    folds: [(学習の打ち切り日, 予測の開始日, 終了日), …] を渡すと、時系列外の予測も保存する（合成器の推定用）。
    """
    out = ROOT / "data/models" / tag
    df = load_frame(hi=end)
    report = dict(training_start=TRAIN_START, training_end_inclusive=end,
                  sealed_period_used_for="training only (never evaluated)", blend_cl=BLEND_CL, seeds=list(SEEDS),
                  refit_factor=REFIT_FACTOR, early_stop_days=EARLY_STOP_DAYS, folds=[])
    model = RaceModelV4().fit(df)
    report["train_races"], report["iterations"] = model.train_races, model.early_stop_iterations
    out.mkdir(parents=True, exist_ok=True)
    report["model_sha256"] = save_model(model, out / "model.pkl")
    oos = []
    for train_end, lo, hi in folds:
        tr = df.filter(pl.col("date") < train_end)
        te = df.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        p1, joint, tri = RaceModelV4().fit(tr, model.early_stop_iterations).predict(te)
        oos.append(oos_rows(te, p1, joint, tri, train_end))
        fold_report, _ = compare(te, p1, joint, tri)
        report["folds"].append(dict(trained_before=train_end, train_races=tr.height // 6, **fold_report))
        print(json.dumps(report["folds"][-1], ensure_ascii=False), flush=True)
    if oos:
        pl.concat(oos).write_parquet(out / "oos_predictions.parquet")
        report["oos_sha256"] = hashlib.sha256((out / "oos_predictions.parquet").read_bytes()).hexdigest()
    report["versions"] = {"lightgbm": lgb.__version__, "polars": pl.__version__, "numpy": np.__version__}
    report["recorded_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (out / "metadata.json").write_text(text)
    (ROOT / f"prereg/{tag}_train.json").write_text(text)
    print(f"Saved {tag}: trained on {model.train_races} races through {model.training_through}")


def oos_rows(te, p1, joint, tri, trained_before):
    keys = te.select(KEY).unique(maintain_order=True)
    return keys.with_columns(
        p1=pl.Series(p1.tolist()), joint=pl.Series(joint.tolist()), tri=pl.Series(tri.tolist()),
        rank=pl.Series(te["rank"].to_numpy().reshape(-1, 6).tolist()), model_trained_before=pl.lit(trained_before))


def oos(tag, lo, hi):
    """保存済みモデルで、学習の打ち切りより後の区間を予測して保存する（合成器の推定用の時系列外予測）。"""
    out = ROOT / "data/models" / tag
    model = load_model(out / "model.pkl")
    if lo <= model.training_through:
        raise ValueError("The window overlaps the training data")
    te = load_frame(lo=lo, hi=hi)
    after = (datetime.strptime(model.training_through, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    path = out / f"oos_{lo}_{hi}.parquet"
    oos_rows(te, *model.predict(te), after).write_parquet(path)
    print(f"{path}: {te.height // 6} races")
    return path


def live_frame(inputs, entries=None, results=None, meta=None, meta4=None):
    """スクショ転記（展示タイム）＋ 当日の公式 B（出走表）＋ 前日までの結果 → 特徴量＋レーティング。"""
    if entries is None or results is None or meta is None:
        entries, results, meta = history()
    frame = v3.live_frame(inputs, entries, results, meta)
    first = inputs["date"].min()
    # 対象日以降の結果は使わない（レーティングは日単位の逐次更新なので、渡さなければ入らない）
    return add_ratings(frame, entries.filter(pl.col("date") <= inputs["date"].max()),
                       results.filter(pl.col("date") < first), meta4)


def probability_rows(keys, p1, joint, tri):
    """予測を縦持ちにする。pool: 1t（1着）/ 2t / 2f / 3t / 3f。艇番号は1始まり。"""
    import live_market_v2 as m2
    blocks = []
    quinella, trio = m2.quinella_from_exacta(joint), trio_from_trifecta(tri)
    spec = [("1t", [(b,) for b in range(6)], p1), ("2t", [tuple(p) for p in PAIRS], joint),
            ("2f", [(a - 1, b - 1) for a, b in m2.POOLS["2f"]], quinella),
            ("3t", [tuple(t) for t in TRIPLES], tri), ("3f", [tuple(t) for t in TRIOS], trio)]
    for pool, combos, p in spec:
        n = len(combos)
        blocks.append(pl.DataFrame({
            **{c: np.repeat(keys[c].to_numpy(), n) for c in KEY},
            "pool": [pool] * (keys.height * n),
            "combo": ["-".join(str(b + 1) for b in combo) for combo in combos] * keys.height,
            "p": p.ravel()}))
    return pl.concat(blocks)


def predict(path, out, model_path=OUT / "model.pkl"):
    inputs = v2.validate_input(pl.read_csv(path, schema_overrides={"date": pl.String}))
    model = load_model(model_path)
    if inputs["date"].min() <= model.training_through:
        raise ValueError("Prediction date must be after model training cutoff")
    if out.exists():
        raise ValueError("Refusing to overwrite prediction")
    target = live_frame(inputs)
    p1, joint, tri = model.predict(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    probability_rows(target.select(KEY).unique(maintain_order=True), p1, joint, tri).write_csv(out)
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    out.with_suffix(".json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v4",
        training_through=model.training_through,
        history_through=results.filter(pl.col("date") < inputs["date"].min())["date"].max(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        note="Odds not used. Entries come from the official B file; history and ratings use results strictly "
             "before the race date. history_through shows how stale it is."), indent=2) + "\n")
    return p1, joint, tri


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "dev", "train", "oos", "predict"])
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--end", default=END)
    ap.add_argument("--tag", default="live_v4")
    ap.add_argument("--fold", nargs=3, action="append", metavar=("TRAIN_BEFORE", "LO", "HI"), default=[])
    ap.add_argument("--lo")
    ap.add_argument("--hi")
    args = ap.parse_args()
    if args.command == "features":
        build_features()
    elif args.command == "dev":
        dev()
    elif args.command == "train":
        train(args.end, args.tag, [tuple(f) for f in args.fold])
    elif args.command == "oos":
        oos(args.tag, args.lo, args.hi)
    elif args.input and args.out:
        predict(args.input, args.out)
    else:
        ap.error("predict requires --input and --out")
