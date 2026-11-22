"""v5 — 凍結済み v4 の確率に、スタート展示の進入コースと展示STを足す補正層。

v4（live_v4.py・live_market_v4.py、data/models/live_v4*/）は書き換えない。v4 の3段（1着 → 2着|1着 →
3着|1・2着）それぞれに、展示の情報だけを使う条件付きロジットの項を足す:

    P'(i)       ∝ P(i)       · exp(x_i · β1)
    P'(j | w)   ∝ P(j | w)   · exp(x_j(w) · β2)
    P'(k | w,s) ∝ P(k | w,s) · exp(x_k(w) · β3)

x は展示の進入が枠なりなら（展示STを使わない版では）全てゼロになるので、そのときは v4 と同じ確率を返す。
展示の進入・STは boatracecsv の stt（締切約8.6分前の撮影。od2 の板と同じ時刻）、運用では入力CSVの
exhibition_course・exhibition_st 列。

なぜ: v4 は展示で進入が変わったレース（全体の約13%）で確定市場に負けている。2026-07-19〜11-12 の
時系列外予測で 3連単 v4 − 市場 は 枠なり −0.007 / 変更あり +0.034、紙上回収率（2連単）は 104.9% / 87.5%。
市場は展示の進入を見ているがモデルは見ていない（設計と事前登録は prereg/live_v5_design.md）。

  evaluate  2026-01-01〜07-18 の時系列外予測（live_v4_y25）で係数を推定し、2026-07-19〜11-12（live_v4a）で
            v4 と比べる。1回だけ（結果ファイルがあれば止まる）
  fit       最終係数を 2026-01-01〜11-12 の時系列外予測全部で推定して data/models/live_v5/ に保存する
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

import evaluate_prospective_v2 as ep2
import live_v4 as f4

ROOT, KEY = f4.ROOT, f4.KEY
OUT = ROOT / "data/models/live_v5"
BOATS = np.arange(1, 7)
PAIR_INDEX = np.full((6, 6), -1)
PAIR_INDEX[f4.PAIRS[:, 0], f4.PAIRS[:, 1]] = np.arange(len(f4.PAIRS))
TRIPLE_INDEX = np.full((6, 6, 6), -1)
TRIPLE_INDEX[f4.TRIPLES[:, 0], f4.TRIPLES[:, 1], f4.TRIPLES[:, 2]] = np.arange(len(f4.TRIPLES))
ENTRY_NAMES = [f"course{c}" for c in range(2, 7)] + ["moved_in", "moved_out", "deep_c1", "deep_c23"]
ST_NAMES = ["st_rel", "st_flying"]
NEXT_NAMES = ["adjacent_to_winner", "outside_winner"]
RIDGE = 1.0
ST_CLIP = (-0.10, 0.40)
DEEP_CAP = 4

FOLD_OOS = ROOT / "data/models/live_v4_y25/oos_20260101_20260718.parquet"
EVAL_OOS = [ROOT / "data/models/live_v4a/oos_predictions.parquet",
            ROOT / "data/models/live_v4a/oos_20260910_20261112.parquet"]
FIT_WINDOW = ("20260101", "20260718")
EVAL_WINDOW = ("20260719", "20261112")


# ---------------------------------------------------------------- 展示の入力
def exhibition(lo, hi):
    """stt（スタート展示）を締切前60分以内の撮影だけ。列: KEY + c1..c6（コース）+ s1..s6（展示ST）。"""
    stt = ep2.raw_previews("stt", lo, hi).filter(
        pl.col("minutes_before_scheduled_deadline").is_between(0, 60, closed="right"))
    return stt.select(*KEY, *[pl.col(f"艇{b}_コース").cast(pl.Int64, strict=False).alias(f"c{b}") for b in BOATS],
                      *[pl.col(f"艇{b}_スタート展示").cast(pl.Float64, strict=False).alias(f"s{b}") for b in BOATS])


def from_inputs(inputs):
    """運用の入力CSV（1艇1行、exhibition_course・exhibition_st）→ exhibition() と同じ横持ち。"""
    wide = inputs.select(*KEY, "boat", pl.col("exhibition_course").cast(pl.Int64, strict=False).alias("c"),
                         pl.col("exhibition_st").cast(pl.Float64, strict=False).alias("s"))
    return wide.pivot(on="boat", index=KEY, values=["c", "s"]).rename(
        {f"{v}_{b}": f"{v}{b}" for v in "cs" for b in BOATS})


def arrays(keys, ex):
    """keys の順に (コース, 展示ST)。コースが 1〜6 の並べ替えでないレースは枠なり・ST欠測として扱う（補正なし）。"""
    k = keys.join(ex, on=KEY, how="left", maintain_order="left")
    course = np.column_stack([k[f"c{b}"].to_numpy() for b in BOATS]).astype(float)
    st = np.column_stack([k[f"s{b}"].to_numpy() for b in BOATS]).astype(float)
    ok = np.isfinite(course).all(axis=1)
    ok[ok] = (np.sort(course[ok], axis=1) == BOATS).all(axis=1)
    course = np.where(ok[:, None], course, BOATS).astype(int)
    st[~(ok & np.isfinite(st).all(axis=1))] = np.nan
    return course, st


# ---------------------------------------------------------------- 特徴量
def boat_features(course, st, use_st, extra=None):
    """(レース, 6艇, 列)。進入が枠なりなら進入の列は全てゼロ。extra (レース, 6艇, k) は末尾に足す（v5b）。"""
    shift = ((course[..., None] == np.arange(2, 7)).astype(float)
             - (BOATS[:, None] == np.arange(2, 7)).astype(float))
    moved_in = np.maximum(BOATS - course, 0).astype(float)
    moved_out = np.maximum(course - BOATS, 0).astype(float)
    deep = np.minimum(moved_in.sum(axis=1, keepdims=True), DEEP_CAP)
    cols = [shift, moved_in[..., None], moved_out[..., None],
            ((course == 1) * deep)[..., None], (np.isin(course, (2, 3)) * deep)[..., None]]
    if use_st:
        has = np.isfinite(st).all(axis=1, keepdims=True)
        clipped = np.clip(np.nan_to_num(st), *ST_CLIP)
        rel = np.where(has, clipped - clipped.mean(axis=1, keepdims=True), 0.0)
        flying = np.where(has, np.nan_to_num(st) < 0, False).astype(float)
        cols += [rel[..., None], flying[..., None]]
    if extra is not None:
        cols.append(extra)
    return np.concatenate(cols, axis=-1)


def next_features(course, base):
    """(レース, 1着艇 w, 候補艇, 列)。base に「1着艇の隣か」「1着艇のすぐ外か」の枠なりとの差を足す。"""
    n = len(course)
    cw = course[:, :, None]
    cj = course[:, None, :]
    bw, bj = BOATS[None, :, None], BOATS[None, None, :]
    adjacent = (np.abs(cj - cw) == 1).astype(float) - (np.abs(bj - bw) == 1).astype(float)
    outside = (cj == cw + 1).astype(float) - (bj == bw + 1).astype(float)
    extra = np.stack([np.broadcast_to(adjacent, (n, 6, 6)), np.broadcast_to(outside, (n, 6, 6))], axis=-1)
    return np.concatenate([np.broadcast_to(base[:, None], (n, 6, 6, base.shape[-1])), extra], axis=-1)


def names(use_st, stage, extra_names=()):
    cols = ENTRY_NAMES + (ST_NAMES if use_st else []) + list(extra_names)
    return cols + (NEXT_NAMES if stage > 1 else [])


# ---------------------------------------------------------------- 3段の分解と合成
def stages(p1, joint, tri):
    """v4 の出力 → (1着, 2着|1着 (n,6,6), 3着|1・2着 (n,6,6,6))。ありえない組は 0。"""
    n = len(p1)
    jm = np.zeros((n, 6, 6))
    jm[:, f4.PAIRS[:, 0], f4.PAIRS[:, 1]] = joint
    tm = np.zeros((n, 6, 6, 6))
    tm[:, f4.TRIPLES[:, 0], f4.TRIPLES[:, 1], f4.TRIPLES[:, 2]] = tri
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.nan_to_num(jm / jm.sum(axis=2, keepdims=True))
        c3 = np.nan_to_num(tm / tm.sum(axis=3, keepdims=True))
    return p1 / p1.sum(axis=1, keepdims=True), c2, c3


def _softmax_offset(offset, score):
    """offset が 0 の候補は除外（確率 0 のまま）。候補が1つも無い行（ありえない組）は全て 0。"""
    valid = offset > 0
    logit = np.where(valid, np.log(np.where(valid, offset, 1.0)) + score, -np.inf)
    top = logit.max(axis=-1, keepdims=True)
    e = np.where(valid, np.exp(logit - np.where(np.isfinite(top), top, 0.0)), 0.0)
    z = e.sum(axis=-1, keepdims=True)
    return np.divide(e, z, out=np.zeros_like(e), where=z > 0)


def correct(p1, joint, tri, course, st, coef, extra=None):
    """補正後の (p1, joint, tri)。coef = {"use_st": bool, "beta": [β1, β2, β3], "extra": [列名…]（v5b のみ）}。"""
    use_st = coef["use_st"]
    if bool(coef.get("extra")) != (extra is not None):
        raise ValueError("Extra features must match the coefficients")
    b1, b2, b3 = (np.asarray(b) for b in coef["beta"])
    s1, c2, c3 = stages(p1, joint, tri)
    base = boat_features(course, st, use_st, extra)
    nxt = next_features(course, base)
    q1 = _softmax_offset(s1, base @ b1)
    q2 = _softmax_offset(c2, nxt @ b2)                                  # (n, w, j)
    q3 = _softmax_offset(c3, (nxt @ b3)[:, :, None, :])                 # (n, w, s, k)
    jm = q1[:, :, None] * q2
    tm = jm[..., None] * q3
    return (q1, jm[:, f4.PAIRS[:, 0], f4.PAIRS[:, 1]],
            tm[:, f4.TRIPLES[:, 0], f4.TRIPLES[:, 1], f4.TRIPLES[:, 2]])


# ---------------------------------------------------------------- 推定
def _fit_stage(offset, x, y, ridge=RIDGE):
    """条件付きロジット（offset つき）。offset (n,m)、x (n,m,k)、y (n,)。リッジ付きの最尤。"""
    rows = np.arange(len(y))
    if not (offset[rows, y] > 0).all():
        raise ValueError("Observed outcome has zero base probability")
    logoff = np.where(offset > 0, np.log(np.where(offset > 0, offset, 1.0)), -np.inf)

    def nll(beta):
        s = logoff + x @ beta
        m = s.max(axis=1, keepdims=True)
        e = np.exp(s - m)
        z = e.sum(axis=1, keepdims=True)
        p = e / z
        value = -(s[rows, y] - m[:, 0] - np.log(z[:, 0])).sum() + ridge * beta @ beta / 2
        g = np.einsum("nm,nmk->k", p, x) - x[rows, y].sum(axis=0) + ridge * beta
        return value, g

    res = minimize(nll, np.zeros(x.shape[-1]), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Fit failed: {res.message}")
    return res.x


def fit(p1, joint, tri, order, course, st, use_st, extra=None, extra_names=()):
    """実際の着順 order (n,6: 1着,2着,3着,…の艇 0始まり) で3段の係数を推定する。"""
    s1, c2, c3 = stages(p1, joint, tri)
    base = boat_features(course, st, use_st, extra)
    nxt = next_features(course, base)
    r = np.arange(len(order))
    w, s, k = order[:, 0], order[:, 1], order[:, 2]
    beta = [_fit_stage(s1, base, w),
            _fit_stage(c2[r, w], nxt[r, w], s),
            _fit_stage(c3[r, w, s], nxt[r, w], k)]
    out = dict(use_st=use_st, beta=[b.tolist() for b in beta],
               names=[names(use_st, k, extra_names) for k in (1, 2, 3)], races=int(len(order)))
    if extra is not None:
        out["extra"] = list(extra_names)
    return out


# ---------------------------------------------------------------- 入出力
def load_oos(paths, lo, hi):
    oos = pl.concat([pl.read_parquet(p) for p in paths]).filter(pl.col("date").is_between(pl.lit(lo), pl.lit(hi))).sort(KEY)
    if oos.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate races in OOS files")
    if (oos["model_trained_before"] > oos["date"]).any():
        raise ValueError("OOS predictions include in-sample rows")
    p1, joint, tri = (np.array(oos[c].to_list()) for c in ("p1", "joint", "tri"))
    order = np.argsort(np.array(oos["rank"].to_list()), axis=1)
    return oos.select(KEY), p1, joint, tri, order


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_fit():
    """最終係数: 2026-01-01〜11-12 の時系列外予測（live_v4_y25 + live_v4a の2本）全部で推定する。"""
    keys, p1, joint, tri, order = load_oos([FOLD_OOS] + EVAL_OOS, FIT_WINDOW[0], EVAL_WINDOW[1])
    course, st = arrays(keys, exhibition(FIT_WINDOW[0], EVAL_WINDOW[1]))
    decision = json.loads((ROOT / "prereg/live_v5_check_result.json").read_text())["decision"]
    if decision["v5"] is None:
        raise SystemExit("The registered check did not adopt v5")
    coef = fit(p1, joint, tri, order, course, st, decision["v5"] == "entry_st")
    v4_model = ROOT / "data/models/live_v4/model.pkl"
    coef.update(fitted_on=[str(keys["date"].min()), str(keys["date"].max())],
                oos_sha256={str(Path(p).relative_to(ROOT)): sha(p) for p in [FOLD_OOS] + EVAL_OOS},
                v4_model_sha256=sha(v4_model), generated_at=datetime.now(timezone.utc).isoformat())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entry_coefficients.json").write_text(json.dumps(coef, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(coef, ensure_ascii=False, indent=2))
    return coef


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["evaluate", "fit"])
    ap.add_argument("--rehearsal", help="配線の試運転の出力先（係数ゼロ。判定に使わない）")
    args = ap.parse_args()
    if args.command == "evaluate":
        import evaluate_v5_entry
        evaluate_v5_entry.run(args.rehearsal)
    else:
        final_fit()
