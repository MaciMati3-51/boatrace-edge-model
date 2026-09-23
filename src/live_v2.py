"""直前情報モデル v2 — 全履歴で学習し、直前情報からは展示タイムだけを使う。

v1（live_model.py）は直前情報CSVが揃う 20260719〜 の 4,224レースだけで学習していた。
1着段は早期停止で木29本しか育たず、直前情報なしで約20万レース学習したモデルにも負けていた
（同じ評価2,717レースで 1着 log-loss 1.2365 vs 1.1910）。K文書の展示タイムは直前情報CSVの
展示タイムと 209,239艇で完全に一致するので、展示タイムは過去全体で学習できる。

- 学習: 20220901〜 の正常レース（6艇完走・着順1〜6）。封印期間も学習には使う。評価には使わない
  （predict_day.py と同じ扱い。封印が止めるのは集計値を見て選ぶことで、学習ではない）
- 1着段: レース単位6クラス softmax（v1 と同じ）
- 2着段: 1着艇を入力に与えたレース単位6クラス softmax。v1 の「候補艇ごとの二値分類＋5艇で正規化」は
  候補艇と1着艇しか見ておらず、残り4艇を見ていなかった（1着段で直した欠陥と同じ形）
- 展示ST・展示進入・チルト・調整重量・気象は使わない。v2 の上に線形の条件付きロジットで足しても
  差は 1着 +0.0003 / 2連単 +0.0016 / 2連複 +0.0005、95%区間は全て0を跨いだ

学習: OMP_NUM_THREADS=6 .venv/bin/python src/live_v2.py train
予測: OMP_NUM_THREADS=6 .venv/bin/python src/live_v2.py predict --input race.csv --out pred.csv
入力は v1 と同じスクショ転記CSVでよい。使う列は出走表10列・展示タイム・取得時刻・締切時刻だけ。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import sklearn

from build_features import racer_history, venue_boat_history
import live_model as v1
import predict_day as baseline

ROOT = Path(__file__).resolve().parents[1]
KEY = baseline.KEY
BK = KEY + ["boat"]
OUT = ROOT / "data/models/live_v2"
TENJI = ["tenji", "tenji_rel", "tenji_rank"]
FEATURES = baseline.FEATURES + TENJI
PER_BOAT = [c for c in FEATURES if c not in ("boat", "jcd")]
ENTRY = v1.ENTRY
REQUIRED = BK + ENTRY + ["tenji", "tkz_captured_at", "deadline_at"]
PAIRS = baseline.PAIRS
TRAIN_START, END = "20220901", "20260909"
# 月次 walk-forward の時系列外予測。合成器の推定（live_market_v2.py）と v1 との比較に使う。
OOS_FOLDS = [("20260719", "20260719", "20260819"), ("20260820", "20260820", "20260909")]


def derive(df):
    # 展示タイムはレース内の相対値が効く。順位は同タイムを平均順位にする（v1 の derive と同じ）
    return df.with_columns(
        tenji_rel=pl.col("tenji") - pl.col("tenji").mean().over(KEY),
        tenji_rank=pl.col("tenji").rank("average").over(KEY),
    )


def build_frame(end=END):
    """学習用の1艇1行。履歴は日単位で1日ずらした expanding（同日の結果は見えない）。"""
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").filter(
        pl.col("date") <= end).with_columns(
        class_ord=pl.col("class").replace_strict(baseline.CLASS_ORD, default=None))
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet").filter(pl.col("date") <= end)
    historic = results.with_columns(
        st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64))
    rh, vb = racer_history(historic), venue_boat_history(historic)
    labels = results.select(*BK, "racer_id", "tenji",
                            rank=pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8))
    df = (entries.join(labels, on=BK + ["racer_id"], how="inner", validate="1:1")
          .join(rh, on=["racer_id", "date"], how="left", validate="m:1")
          .join(vb, on=["jcd", "boat", "date"], how="left", validate="m:1"))
    normal = (df.group_by(KEY)
              .agg(n=pl.len(), ranked=pl.col("rank").is_not_null().sum(), distinct=pl.col("rank").n_unique())
              .filter((pl.col("n") == 6) & (pl.col("ranked") == 6) & (pl.col("distinct") == 6))
              .select(KEY))
    return derive(df.join(normal, on=KEY).filter(pl.col("date") >= TRAIN_START).sort(BK))


class RaceModelV2:
    def __init__(self):
        self.features = FEATURES
        self.jcd = FEATURES.index("jcd")
        self.per_boat = [FEATURES.index(c) for c in PER_BOAT]

    def matrix(self, df):
        return df.select(self.features).to_numpy().astype(float).reshape(-1, 6, len(self.features))

    def first_matrix(self, x):
        return np.column_stack([x[:, 0, self.jcd], x[:, :, self.per_boat].reshape(len(x), -1)])

    def second_matrix(self, x, winner):
        # [場, 1着艇(0..5), 6艇ぶんの特徴]。1着艇の番号をカテゴリとして渡す
        return np.column_stack([x[:, 0, self.jcd], winner, x[:, :, self.per_boat].reshape(len(x), -1)])

    def fit(self, df):
        x = self.matrix(df)
        rank = df["rank"].to_numpy().reshape(-1, 6)
        first, second = rank.argmin(axis=1), np.argsort(rank, axis=1)[:, 1]
        self.first = baseline.model([0]).fit(self.first_matrix(x), first)
        self.second = baseline.model([0, 1]).fit(self.second_matrix(x, first), second)
        for clf in (self.first, self.second):
            if list(clf.classes_) != list(range(6)):
                raise ValueError("Model did not learn all six classes")
        self.training_through = df["date"].max()
        return self

    def predict(self, df):
        x = self.matrix(df)
        n = len(x)
        p1 = self.first.predict_proba(self.first_matrix(x))
        conditional = np.zeros((n, 6, 6))
        for w in range(6):
            p = self.second.predict_proba(self.second_matrix(x, np.full(n, w)))
            p[:, w] = 0  # 1着艇は2着になれない
            conditional[:, w] = p / p.sum(axis=1, keepdims=True)
        joint = p1[:, PAIRS[:, 0]] * conditional[:, PAIRS[:, 0], PAIRS[:, 1]]
        if not np.isfinite(joint).all() or not np.allclose(joint.sum(axis=1), 1):
            raise ValueError("Invalid probabilities")
        return p1, joint


def save_model(model, path):
    payload = pickle.dumps(dict(version="live_v2", first=model.first, second=model.second,
                                features=model.features, training_through=model.training_through))
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path):
    # ローカルで生成したものだけを読む。pickle は実行可能データ
    with path.open("rb") as f:
        saved = pickle.load(f)
    if saved.get("version") != "live_v2" or saved["features"] != FEATURES:
        raise ValueError("Not a live_v2 artifact or feature list changed")
    model = RaceModelV2()
    model.first, model.second = saved["first"], saved["second"]
    model.training_through = saved["training_through"]
    return model


def day_bootstrap(diff, dates, n=2000):
    """日単位の対応付きブートストラップ（レースの重みは保つ）。diff は (レース, 指標)。"""
    unique = np.unique(dates)
    sums = np.array([diff[dates == d].sum(axis=0) for d in unique])
    counts = np.array([(dates == d).sum() for d in unique])
    picks = np.random.default_rng(0).integers(0, len(unique), (n, len(unique)))
    boots = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None]
    return np.quantile(boots, [.025, .975], axis=0).T.tolist()


def v1_probabilities(keys, name):
    """live_v1 が保存した評価期間の確率を keys の順に並べる。"""
    p = keys.join(pl.read_parquet(v1.OUT / f"{name}_evaluation.parquet"), on=KEY, how="left",
                  maintain_order="left")
    if p.height != keys.height * 30 or p["p_2t"].null_count():
        raise ValueError(f"live_v1 {name} probabilities do not cover the comparison races")
    p = p.sort([pl.col(c) for c in KEY] + ["first", "second"], maintain_order=True)
    joint = p["p_2t"].to_numpy().reshape(-1, 30)
    return p["p_1st"].to_numpy().reshape(-1, 30)[:, ::5], joint


def train():
    df = build_frame()
    report = dict(training_start=TRAIN_START, training_end_inclusive=END, features=FEATURES,
                  second_stage="race-level 6-class softmax given the winner",
                  sealed_period_used_for="training only (never evaluated)")
    folds = []
    for train_end, lo, hi in OOS_FOLDS:
        tr = df.filter(pl.col("date") < train_end)
        te = df.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        model = RaceModelV2().fit(tr)
        p1, joint = model.predict(te)
        keys = te.select(KEY).unique(maintain_order=True)
        folds.append(keys.with_columns(
            p1=pl.Series(p1.tolist()), joint=pl.Series(joint.tolist()),
            rank=pl.Series(te["rank"].to_numpy().reshape(-1, 6).tolist()),
            model_trained_before=pl.lit(train_end)))
        report.setdefault("folds", []).append(dict(
            trained_before=train_end, predicted=[lo, hi], train_races=tr.height // 6,
            predicted_races=keys.height, iterations=[int(model.first.n_iter_), int(model.second.n_iter_)]))
        print(report["folds"][-1], flush=True)
    oos = pl.concat(folds)
    OUT.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(OUT / "oos_predictions.parquet")

    # v1 と同じ評価レース（v1 の学習末尾 20260819 の翌日から）で比べる
    keys = pl.read_parquet(v1.OUT / "preview_evaluation.parquet").select(KEY).unique().sort(KEY)
    te = df.join(keys, on=KEY).sort(BK)
    ours = oos.join(keys, on=KEY).sort(KEY)
    if ours.height != keys.height or te.height != keys.height * 6:
        raise ValueError("v2 predictions do not cover the live_v1 evaluation races")
    losses = {"v2": v1.losses(te, np.array(ours["p1"].to_list()), np.array(ours["joint"].to_list()))}
    for name in ("baseline", "preview"):
        losses[f"v1_{name}"] = v1.losses(te, *v1_probabilities(keys, name))
    dates = keys["date"].to_numpy()
    report["comparison"] = dict(
        races=keys.height, period=[keys["date"].min(), keys["date"].max()],
        order=["1着", "2連単", "2連複"], negative_is_better=True,
        logloss={k: v.mean(axis=0).tolist() for k, v in losses.items()},
        v2_minus_v1_preview=dict(mean=(losses["v2"] - losses["v1_preview"]).mean(axis=0).tolist(),
                                 day_bootstrap_95ci=day_bootstrap(losses["v2"] - losses["v1_preview"], dates)),
        status="exploratory: this period was already used to develop live_v1 and to choose the v2 design")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2), flush=True)

    model = RaceModelV2().fit(df)
    report["train_races"] = df.height // 6
    report["iterations"] = [int(model.first.n_iter_), int(model.second.n_iter_)]
    report["model_sha256"] = save_model(model, OUT / "model.pkl")
    report["oos_sha256"] = hashlib.sha256((OUT / "oos_predictions.parquet").read_bytes()).hexdigest()
    report["versions"] = {"sklearn": sklearn.__version__, "polars": pl.__version__, "numpy": np.__version__}
    report["recorded_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (OUT / "metadata.json").write_text(text)
    (ROOT / "prereg/live_v2_comparison.json").write_text(text)
    print(f"Saved model trained on {report['train_races']} races through {model.training_through}")


def validate_input(df):
    """スクショ転記の検査。v1 の列が余分にあってもよい。"""
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if not df.height or df.unique(BK).height != df.height:
        raise ValueError("Empty input or duplicate race/boat")
    for group in df.partition_by(KEY):
        if sorted(group["boat"].to_list()) != list(range(1, 7)):
            raise ValueError("Each race must contain boats 1..6 exactly once")
        if group["tkz_captured_at"].n_unique() != 1 or group["deadline_at"].n_unique() != 1:
            raise ValueError("Inconsistent timestamp within a race")
        deadline = datetime.fromisoformat(group["deadline_at"][0])
        stamp = datetime.fromisoformat(group["tkz_captured_at"][0])
        if deadline.utcoffset() is None or stamp.utcoffset() is None:
            raise ValueError("Timezone required")
        if deadline.strftime("%Y%m%d") != group["date"][0]:
            raise ValueError("Deadline date does not match race date")
        if not 0 < (deadline - stamp).total_seconds() <= 3600:
            raise ValueError("Exhibition capture outside pre-deadline window")
    values = df.select(ENTRY + ["tenji"]).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite entry or exhibition values")
    if not df["tenji"].is_between(5, 9).all():
        raise ValueError("Exhibition time outside expected bounds")
    if not df["class_ord"].cast(pl.Float64).is_in([1.0, 2.0, 3.0, 4.0]).all():
        raise ValueError("class_ord must be 1..4 (B2..A1)")
    return df.sort(BK)


def with_history(df, results):
    """予測日の前日までの結果だけで履歴を付ける。新人は履歴なし（学習時と同じく null）。"""
    frames = []
    for date in df["date"].unique().sort():
        rh, vb = baseline.history_as_of(results, date)
        frames.append(df.filter(pl.col("date") == date)
                      .join(rh, on="racer_id", how="left", validate="m:1")
                      .join(vb, on=["jcd", "boat"], how="left", validate="m:1"))
    return derive(pl.concat(frames).sort(BK))


def probability_rows(keys, p1, joint):
    rows = []
    lookup = {tuple(pair): i for i, pair in enumerate(PAIRS)}
    for i, key in enumerate(keys.iter_rows(named=True)):
        for j, (a, b) in enumerate(PAIRS):
            rows.append(dict(**key, first=int(a + 1), second=int(b + 1), p_1st=float(p1[i, a]),
                             p_2t=float(joint[i, j]),
                             p_2f=float(joint[i, j] + joint[i, lookup[(b, a)]]) if a < b else None))
    return pl.DataFrame(rows)


def predict(path, out, model_path=OUT / "model.pkl"):
    df = validate_input(pl.read_csv(path, schema_overrides={"date": pl.String}))
    model = load_model(model_path)
    if df["date"].min() <= model.training_through:
        raise ValueError("Prediction date must be after model training cutoff")
    if out.exists():
        raise ValueError("Refusing to overwrite prediction")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    target = with_history(df, results)
    p1, joint = model.predict(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    probability_rows(target.select(KEY).unique(maintain_order=True), p1, joint).write_csv(out)
    before = results.filter(pl.col("date") < df["date"].min())["date"].max()
    out.with_suffix(".json").write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), model="live_v2",
        training_through=model.training_through, history_through=before,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        note="Odds not used. Racer history uses local results strictly before the race date; "
             "history_through shows how stale it is."), indent=2) + "\n")
    return p1, joint


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "train":
        train()
    elif args.input and args.out:
        predict(args.input, args.out)
    else:
        ap.error("predict requires --input and --out")
