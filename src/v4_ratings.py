"""v4 の追加特徴: 相手の強さを引いたレーティング（逐次推定の Plackett-Luce）。

v3 の選手成績（r_win・r_rpts など）は「会場×枠の平均との差」までしか引いていない。同じ3着でも
A1 が5人いるレースと B2 が5人いるレースを区別しない。モーターの成績（m_rpts）も、乗った選手の腕を
引いていない。ここでは 1出走の強さを

    θ = 選手 + 選手×枠(偏差) + 選手×会場(偏差) + モーター(会場×番号×使用期間) + 会場×枠

の和で表し、着順の Plackett-Luce 尤度の勾配で1日ごとに更新する（Elo の多艇版）。更新は
「その日の全レースを前日までの値で評価 → まとめて更新」なので、対象日の特徴量は対象日より前の
結果だけで決まる（v3_features と同じ日単位の規則）。

    F・L（スタート事故）と欠場はそのレースから外す（腕の情報ではない）
    失格・転覆など着順なしで完走しなかった艇は「着順のある全艇に負けた」として扱う

学習率の違う2系統（slow / fast）を独立に回す。fast は直近の調子に寄り、会場×枠も速く動かす
（季節・水面の傾向を拾う）。

出力列（1艇1行、BK で v3 の特徴量に結合する）:
    rt_{s,f}_racer   選手          rt_{s,f}_lane   選手×枠の偏差     rt_{s,f}_home  選手×会場の偏差
    rt_{s,f}_motor   モーター      rt_{s,f}_venue  会場×枠
    rt_{s,f}_theta   合計          rt_{s,f}_p      レース内 softmax（レーティングだけの1着確率）
    rt_d_racer       fast − slow の選手（直近の上り調子・下り調子）、rt_d_motor 同モーター
    rt_n             選手の更新回数、rt_m_n モーターの更新回数
"""
import numpy as np
import polars as pl

import v3_features as feat

KEY, BK = feat.KEY, feat.BK
# 学習率は学習期間の内側（20220101〜20240630）で、レーティングだけの1着 log-loss を見て決めた（v4 設計書 §2）。
# home と fast 側の venue は開発の検証区間で1回だけ試して採用した（1着 −0.0008）
SYSTEMS = {
    "s": dict(racer=0.03, lane=0.02, motor=0.02, venue=0.004, home=0.01, boost=40.0, stage=0.45),
    "f": dict(racer=0.08, lane=0.03, motor=0.05, venue=0.02, home=0.02, boost=40.0, stage=0.45),
}
STAGES = 5
PARTS = ("racer", "lane", "motor", "venue", "home")


def _index(values):
    uniq, inv = np.unique(values, return_inverse=True)
    return uniq, inv


def _prepare(entries, results):
    """番組表（特徴量を出す行）と結果（更新に使う行）を、共通の整数キーに直す。"""
    mp = feat.motor_periods(entries)
    ent = entries.select(*BK, "racer_id", "motor_no").join(mp, on=["jcd", "date"], how="left").sort(BK)
    res = (results.select(*BK, "racer_id", "motor_no", "rank_raw")
           .filter(~pl.col("rank_raw").str.starts_with("K")
                   & ~pl.col("rank_raw").str.starts_with("F") & ~pl.col("rank_raw").str.starts_with("L"))
           .join(mp, on=["jcd", "date"], how="left")
           .with_columns(rank=pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8).fill_null(9))
           .sort(BK))
    both = pl.concat([ent.select("racer_id", "jcd", "motor_no", "mperiod"),
                      res.select("racer_id", "jcd", "motor_no", "mperiod")])
    racers = both["racer_id"].unique().sort()
    motors = (both.filter(pl.col("motor_no").is_not_null() & pl.col("mperiod").is_not_null())
              .select("jcd", "motor_no", "mperiod").unique().sort("jcd", "motor_no", "mperiod")
              .with_row_index("motor_ix"))
    racer_ix = pl.DataFrame({"racer_id": racers}).with_row_index("racer_ix")

    def code(df):
        return (df.join(racer_ix, on="racer_id", how="left")
                .join(motors, on=["jcd", "motor_no", "mperiod"], how="left")
                .with_columns(pl.col("motor_ix").cast(pl.Int64).fill_null(-1), pl.col("racer_ix").cast(pl.Int64)))

    return code(ent), code(res), racers.len(), motors.height


def _slices(dates):
    """日付（昇順に並んだ配列）→ {日付: (開始, 終了)}。"""
    uniq, start = np.unique(dates, return_index=True)
    end = np.append(start[1:], len(dates))
    return {d: (a, b) for d, a, b in zip(uniq.tolist(), start.tolist(), end.tolist())}


def _race_blocks(race_code):
    """レースごとの行を (レース, 6) の添字に詰める。足りない枠は -1。"""
    _, first, counts = np.unique(race_code, return_index=True, return_counts=True)
    order = np.argsort(first)
    first, counts = first[order], counts[order]
    block = np.full((len(first), 6), -1, dtype=np.int64)
    for j in range(6):
        ok = counts > j
        block[ok, j] = first[ok] + j
    return block


class _System:
    def __init__(self, cfg, n_racers, n_motors):
        self.cfg = cfg
        self.racer = np.zeros(n_racers)
        self.lane = np.zeros((n_racers, 6))
        self.motor = np.zeros(n_motors + 1)  # 末尾は「モーター不明」の捨て場（常に 0 に戻す）
        self.venue = np.zeros((25, 6))
        self.home = np.zeros((n_racers, 25))
        self.n = np.zeros(n_racers)
        self.m_n = np.zeros(n_motors + 1)

    def parts(self, racer, lane, motor, jcd):
        return (self.racer[racer], self.lane[racer, lane], self.motor[motor], self.venue[jcd, lane],
                self.home[racer, jcd])

    def update(self, racer, lane, motor, jcd, rank, block):
        """block: (レース, 6) の行添字（-1 は空き）。その日の全レースを前日までの値で評価してから更新する。"""
        c = self.cfg
        valid = block >= 0
        ix = np.where(valid, block, 0)
        theta = sum(self.parts(racer[ix], lane[ix], motor[ix], jcd[ix]))
        rk = np.where(valid, rank[ix], 99)
        alive = valid.copy()
        grad = np.zeros(block.shape)
        rows = np.arange(len(block))
        for stage in range(STAGES):
            pick = np.where(alive, rk, 99).argmin(axis=1)
            # その段で選ばれる艇に着順があり、相手が1艇以上残っているレースだけ
            use = (rk[rows, pick] <= 6) & alive[rows, pick] & (alive.sum(axis=1) > 1)
            if not use.any():
                break
            t = np.where(alive, theta, -np.inf)
            p = np.exp(t - t.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            g = -p
            g[rows, pick] += 1.0
            grad += (c["stage"] ** stage) * g * use[:, None]
            alive[rows[use], pick[use]] = False
        r, ln, m, j, g = racer[ix][valid], lane[ix][valid], motor[ix][valid], jcd[ix][valid], grad[valid]
        # 出走の少ない選手・モーターは速く動かす（1/(1+n/boost) で減衰する上乗せ）
        kr = 1.0 + 4.0 * c["boost"] / (self.n[r] + c["boost"] + 1e-9)
        km = 1.0 + 4.0 * c["boost"] / (self.m_n[m] + c["boost"] + 1e-9)
        np.add.at(self.racer, r, c["racer"] * kr * g)
        np.add.at(self.lane, (r, ln), c["lane"] * g)
        np.add.at(self.motor, m, c["motor"] * km * g)
        np.add.at(self.venue, (j, ln), c["venue"] * g)
        np.add.at(self.home, (r, j), c["home"] * g)
        np.add.at(self.n, r, 1.0)
        np.add.at(self.m_n, m, 1.0)
        self.motor[-1] = 0.0
        self.m_n[-1] = 0.0


def build(entries, results, systems=None):
    """entries の全行にレーティング特徴を付ける。results は使える結果の全て（対象日のものは使われない）。"""
    systems = systems or SYSTEMS
    ent, res, n_racers, n_motors = _prepare(entries, results)
    e = {c: ent[c].to_numpy() for c in ("racer_ix", "motor_ix", "jcd")}
    e["lane"] = ent["boat"].to_numpy() - 1
    r = {c: res[c].to_numpy() for c in ("racer_ix", "motor_ix", "jcd", "rank")}
    r["lane"] = res["boat"].to_numpy() - 1
    e_days, r_days = _slices(ent["date"].to_numpy()), _slices(res["date"].to_numpy())
    r_race = (res["jcd"].to_numpy() * 100 + res["race_no"].to_numpy()).astype(np.int64)
    state = {tag: _System(cfg, n_racers, n_motors) for tag, cfg in systems.items()}
    out = {f"rt_{tag}_{part}": np.zeros(ent.height) for tag in systems for part in PARTS}
    out["rt_n"], out["rt_m_n"] = np.zeros(ent.height), np.zeros(ent.height)
    first = next(iter(state.values()))
    for day in sorted(set(e_days) | set(r_days)):
        if day in e_days:  # 先に特徴量を出す（前日までの状態）
            a, b = e_days[day]
            for tag, s in state.items():
                parts = s.parts(e["racer_ix"][a:b], e["lane"][a:b], e["motor_ix"][a:b], e["jcd"][a:b])
                for part, v in zip(PARTS, parts):
                    out[f"rt_{tag}_{part}"][a:b] = v
            out["rt_n"][a:b] = first.n[e["racer_ix"][a:b]]
            out["rt_m_n"][a:b] = first.m_n[e["motor_ix"][a:b]]
        if day in r_days:  # その日の結果で更新
            a, b = r_days[day]
            block = _race_blocks(r_race[a:b])
            for s in state.values():
                s.update(r["racer_ix"][a:b], r["lane"][a:b], r["motor_ix"][a:b], r["jcd"][a:b], r["rank"][a:b], block)
    df = ent.select(BK).with_columns([pl.Series(k, v) for k, v in out.items()])
    for tag in systems:
        theta = sum(pl.col(f"rt_{tag}_{part}") for part in PARTS)
        df = df.with_columns(theta.alias(f"rt_{tag}_theta"))
        df = df.with_columns(
            (pl.col(f"rt_{tag}_theta").exp() / pl.col(f"rt_{tag}_theta").exp().sum().over(KEY)).alias(f"rt_{tag}_p"))
    if {"s", "f"} <= set(systems):
        df = df.with_columns(rt_d_racer=pl.col("rt_f_racer") - pl.col("rt_s_racer"),
                             rt_d_motor=pl.col("rt_f_motor") - pl.col("rt_s_motor"))
    return df


COLUMNS = ([f"rt_{tag}_{part}" for tag in SYSTEMS for part in PARTS + ("theta", "p")]
           + ["rt_d_racer", "rt_d_motor", "rt_n", "rt_m_n"])
