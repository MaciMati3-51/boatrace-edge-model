"""v3 の特徴量。全て「対象日より前の結果」だけから作る（日単位。同日の別レースは見ない）。

v2 は選手の通算成績5列と会場×枠の1着率だけだった。v3 で足すもの:

  選手        直近90日 / 365日 / 通算 の 1着率・連対率・3連対率・得点・ST・展示タイム傾向
              枠の有利不利を引いた残差（会場×枠の平均との差）
  選手×枠     その枠に入ったときの成績・ST・決まり手（逃げ/差し/まくり/まくり差し）・
              1号艇を逃がす率・前付けの頻度
  モーター    今の使用期間（会場ごとに年1回の交換を出走表から検出）の成績と展示タイム
  ボート      直近120日の成績
  今節        同じ会場での直近7日の成績・展示・ST
  その他      休み明け日数・F からの日数・直近180日の F 回数・登録番号（デビュー時期の代理）
  レース      レース番号・締切時刻・月・レース種別・シリーズ

学習と予測で同じ関数を通す。予測日は results が無いので entries の行だけが対象として入り、
集計値は前日までの結果から決まる。

実行（キャッシュ生成）: .venv/bin/python src/v3_features.py
出力: data/features/v3.parquet  1行 = (date, jcd, race_no, boat)
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
KEY = ["date", "jcd", "race_no"]
BK = KEY + ["boat"]
CLASS_ORD = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
PTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
KIMARITE = {"逃げ": "nige", "差し": "sashi", "まくり": "makuri", "まくり差し": "mzashi"}
RACE_TYPES = {"優勝戦": 5, "準優勝戦": 4, "準優": 4, "予選特賞": 3, "特賞": 3, "予選特選": 3, "特選": 3,
              "選抜": 3, "予選": 2, "一般": 1, "一般戦": 1}
SERIES = {"一般": 0, "ヴィーナス": 1, "オールレディース": 2, "ルーキー": 3, "マスターズ": 4}


def _date(col="date"):
    return pl.col(col).str.strptime(pl.Date, "%Y%m%d")


def events(results):
    """1出走1行。順位・得点・ST・展示の相対値と、レースの決まり手。"""
    rank = pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8)
    ev = results.with_columns(
        rank=rank,
        absent=pl.col("rank_raw").str.starts_with("K"),
        st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64),
        tenji=pl.when(pl.col("tenji") > 0).then(pl.col("tenji")),
    ).filter(~pl.col("absent"))
    ev = ev.with_columns(
        win=(pl.col("rank") == 1).fill_null(False).cast(pl.Float64),
        top2=(pl.col("rank") <= 2).fill_null(False).cast(pl.Float64),
        top3=(pl.col("rank") <= 3).fill_null(False).cast(pl.Float64),
        pts=pl.col("rank").replace_strict(PTS, default=0, return_dtype=pl.Float64).fill_null(0.0),
        F=pl.col("rank_raw").str.starts_with("F").cast(pl.Float64),
        L=pl.col("rank_raw").str.starts_with("L").cast(pl.Float64),
        strel=pl.col("st_num") - pl.col("st_num").mean().over(KEY),
        tenjirel=pl.col("tenji") - pl.col("tenji").mean().over(KEY),
        moved=(pl.col("entry_course") != pl.col("boat")).cast(pl.Float64),
        cdiff=(pl.col("entry_course") - pl.col("boat")).cast(pl.Float64),
        l1win=((pl.col("boat") == 1) & (pl.col("rank") == 1)).any().over(KEY).cast(pl.Float64),
        d=_date(),
    )
    return ev.with_columns(
        *[((pl.col("win") == 1) & (pl.col("kimarite") == k)).fill_null(False).cast(pl.Float64).alias(f"k_{v}")
          for k, v in KIMARITE.items()])


def expanding(daily, keys, cols, suffix):
    """keys ごとに、その日より前の累計。"""
    daily = daily.sort(keys + ["d"])
    return daily.select(*keys, "d", *[(pl.col(c).cum_sum().over(keys) - pl.col(c)).alias(f"{c}{suffix}") for c in cols])


def window(daily, keys, cols, period, suffix):
    """keys ごとに、[その日−period, その日) の合計。"""
    daily = daily.sort(keys + ["d"])
    return daily.rolling("d", period=period, closed="left", group_by=keys).agg(
        [pl.col(c).sum().alias(f"{c}{suffix}") for c in cols])


def daily_sums(ev, targets, keys, sums):
    """keys×日 の合計。対象行（結果がまだ無い日を含む）を 0 として混ぜ、どの対象日にも行があるようにする。"""
    agg = ev.group_by(keys + ["d"]).agg([e.alias(n) for n, e in sums.items()])
    zero = targets.select(keys + ["d"]).unique().with_columns([pl.lit(0.0).alias(n) for n in sums])
    return (pl.concat([agg.with_columns([pl.col(n).cast(pl.Float64) for n in sums]), zero.select(agg.columns)])
            .group_by(keys + ["d"]).agg([pl.col(n).sum() for n in sums]))


def motor_periods(entries):
    """会場ごとのモーター使用期間。交換直後は全モーターの2連率が 0.0 になる。"""
    z = (entries.group_by(["jcd", "date"]).agg(z=(pl.col("motor_2r") == 0).mean())
         .sort(["jcd", "date"]))
    # 開催が90日以上空いたら別の期間として扱う（データの欠け・長期休催をまたいで別のモーターの成績を混ぜない）
    gap = (_date() - _date().shift(1).over("jcd")).dt.total_days().fill_null(0) > 90
    return z.with_columns(
        mperiod=(((pl.col("z") >= 0.9) & (pl.col("z").shift(1).over("jcd").fill_null(0.0) < 0.9)) | gap)
        .cast(pl.Int32).cum_sum().over("jcd")).select("jcd", "date", "mperiod")


def build(entries, results, meta, tenji=None):
    """entries の全行に特徴量を付ける。results は使える結果の全て（対象日のものは集計に入らない）。

    tenji: 対象行の展示タイム (BK + tenji)。省略時は results の値（過去の検証用）。
    """
    ev = events(results)
    tg = entries.with_columns(d=_date(), class_ord=pl.col("class").replace_strict(CLASS_ORD, default=None))
    mp = motor_periods(entries)
    tg = tg.join(mp, on=["jcd", "date"], how="left")
    ev = ev.join(mp, on=["jcd", "date"], how="left")

    # ---- 枠の基準（全国の枠別、会場×枠）: 残差の基準と縮小の事前値 ----
    base_sums = dict(n=pl.len(), win=pl.col("win").sum(), top2=pl.col("top2").sum(), top3=pl.col("top3").sum(),
                     pts=pl.col("pts").sum(), st=pl.col("st_num").sum(), st_n=pl.col("st_num").is_not_null().sum())
    names = list(base_sums)
    nb = expanding(daily_sums(ev, tg, ["boat"], base_sums), ["boat"], names, "")
    nb = nb.select("boat", "d", *[(pl.col(c) / pl.col("n")).alias(f"nb_{c}") for c in ("win", "top2", "top3", "pts")],
                   nb_st=pl.col("st") / pl.col("st_n"))
    vb = expanding(daily_sums(ev, tg, ["jcd", "boat"], base_sums), ["jcd", "boat"], names, "")
    # 会場×枠は全国の枠平均へ 50走ぶん縮小する（データ先頭の日は全国平均も無いので 0 になる）
    vb = vb.join(nb, on=["boat", "d"], how="left").select(
        "jcd", "boat", "d", vb_n=pl.col("n"),
        *[((pl.col(c) + 50 * pl.col(f"nb_{c}").fill_nan(None).fill_null(0.0)) / (pl.col("n") + 50)).alias(f"vb_{c}")
          for c in ("win", "top2", "top3", "pts")])
    ev = ev.join(vb, on=["jcd", "boat", "d"], how="left").with_columns(
        rwin=pl.col("win") - pl.col("vb_win"), rtop3=pl.col("top3") - pl.col("vb_top3"),
        rpts=pl.col("pts") - pl.col("vb_pts"))

    # ---- 選手 ----
    racer_sums = dict(
        n=pl.len(), win=pl.col("win").sum(), top2=pl.col("top2").sum(), top3=pl.col("top3").sum(),
        pts=pl.col("pts").sum(), rwin=pl.col("rwin").sum(), rtop3=pl.col("rtop3").sum(), rpts=pl.col("rpts").sum(),
        st=pl.col("st_num").sum(), st_n=pl.col("st_num").is_not_null().sum(), st_sq=(pl.col("st_num") ** 2).sum(),
        strel=pl.col("strel").sum(), tj=pl.col("tenjirel").sum(), tj_n=pl.col("tenjirel").is_not_null().sum(),
        F=pl.col("F").sum(), L=pl.col("L").sum(), moved=pl.col("moved").sum(), cdiff=pl.col("cdiff").sum(),
        c_n=pl.col("moved").is_not_null().sum())
    rd = daily_sums(ev, tg, ["racer_id"], racer_sums)
    cols = list(racer_sums)
    racer = expanding(rd, ["racer_id"], cols, "_all")
    for period, suffix in (("90d", "_90"), ("365d", "_365")):
        racer = racer.join(window(rd, ["racer_id"], cols[:15], period, suffix), on=["racer_id", "d"])
    racer = racer.join(window(rd, ["racer_id"], ["F"], "180d", "_180"), on=["racer_id", "d"])
    # 休み明け・F からの日数
    last = rd.sort(["racer_id", "d"]).with_columns(
        last_start=pl.when(pl.col("n") > 0).then(pl.col("d")).forward_fill().shift(1).over("racer_id"),
        last_F=pl.when(pl.col("F") > 0).then(pl.col("d")).forward_fill().shift(1).over("racer_id"),
    ).select("racer_id", "d",
             rest_days=(pl.col("d") - pl.col("last_start")).dt.total_days().cast(pl.Float64),
             days_since_F=(pl.col("d") - pl.col("last_F")).dt.total_days().cast(pl.Float64))
    racer = racer.join(last, on=["racer_id", "d"])

    def rates(suffix, k=20.0):
        n = pl.col(f"n{suffix}")
        out = [n.alias(f"r_n{suffix}")]
        # 率は全国の枠平均ではなく 0 に寄る残差と、素の率（事前値=全選手のおおよその平均）を両方持つ
        for c, prior in (("win", 1 / 6), ("top2", 2 / 6), ("top3", 3 / 6), ("pts", 31 / 6)):
            out.append(((pl.col(f"{c}{suffix}") + k * prior) / (n + k)).alias(f"r_{c}{suffix}"))
        for c in ("rwin", "rtop3", "rpts"):
            out.append((pl.col(f"{c}{suffix}") / (n + k)).alias(f"r_{c}{suffix}"))
        out += [(pl.col(f"st{suffix}") / pl.col(f"st_n{suffix}")).alias(f"r_st{suffix}"),
                (pl.col(f"strel{suffix}") / (pl.col(f"st_n{suffix}") + 10)).alias(f"r_strel{suffix}"),
                (pl.col(f"tj{suffix}") / (pl.col(f"tj_n{suffix}") + 10)).alias(f"r_tj{suffix}")]
        return out

    a = "_all"
    racer = racer.select(
        "racer_id", "d", *rates("_90"), *rates("_365"), *rates("_all"),
        r_st_sd=((pl.col(f"st_sq{a}") / pl.col(f"st_n{a}")) - (pl.col(f"st{a}") / pl.col(f"st_n{a}")) ** 2)
        .clip(0).sqrt(),
        r_F_rate=pl.col(f"F{a}") / (pl.col(f"n{a}") + 50), r_L_rate=pl.col(f"L{a}") / (pl.col(f"n{a}") + 50),
        r_moved=pl.col(f"moved{a}") / (pl.col(f"c_n{a}") + 20), r_cdiff=pl.col(f"cdiff{a}") / (pl.col(f"c_n{a}") + 20),
        r_F_180=pl.col("F_180"), rest_days=pl.col("rest_days"), days_since_F=pl.col("days_since_F"))

    # ---- 選手×枠 ----
    lane_sums = dict(
        n=pl.len(), win=pl.col("win").sum(), top2=pl.col("top2").sum(), top3=pl.col("top3").sum(),
        st=pl.col("st_num").sum(), st_n=pl.col("st_num").is_not_null().sum(), strel=pl.col("strel").sum(),
        moved=pl.col("moved").sum(), c_n=pl.col("moved").is_not_null().sum(), l1win=pl.col("l1win").sum(),
        **{f"k_{v}": pl.col(f"k_{v}").sum() for v in KIMARITE.values()})
    ld = daily_sums(ev, tg, ["racer_id", "boat"], lane_sums)
    lane = expanding(ld, ["racer_id", "boat"], list(lane_sums), "").join(
        window(ld, ["racer_id", "boat"], ["n", "win", "top2", "top3"], "365d", "_365"), on=["racer_id", "boat", "d"])
    lane = lane.join(nb, on=["boat", "d"], how="left")
    k = 15.0
    lane = lane.select(
        "racer_id", "boat", "d", l_n=pl.col("n"),
        *[((pl.col(c) + k * pl.col(f"nb_{c}").fill_nan(None).fill_null(0.0)) / (pl.col("n") + k)).alias(f"l_{c}")
          for c in ("win", "top2", "top3")],
        *[((pl.col(f"{c}_365") + k * pl.col(f"nb_{c}").fill_nan(None).fill_null(0.0)) / (pl.col("n_365") + k)).alias(f"l_{c}_365")
          for c in ("win", "top2", "top3")],
        l_n_365=pl.col("n_365"),
        l_st=(pl.col("st") + 5 * pl.col("nb_st").fill_nan(None).fill_null(0.17)) / (pl.col("st_n") + 5),
        l_strel=pl.col("strel") / (pl.col("st_n") + 10),
        l_moved=pl.col("moved") / (pl.col("c_n") + 10),
        l_l1win=pl.col("l1win") / (pl.col("n") + k),
        *[(pl.col(f"k_{v}") / (pl.col("n") + k)).alias(f"l_{v}") for v in KIMARITE.values()])

    # ---- モーター（使用期間内）・ボート ----
    motor_sums = dict(n=pl.len(), top2=pl.col("top2").sum(), top3=pl.col("top3").sum(), rpts=pl.col("rpts").sum(),
                      rwin=pl.col("rwin").sum(), tj=pl.col("tenjirel").sum(),
                      tj_n=pl.col("tenjirel").is_not_null().sum(), strel=pl.col("strel").sum(),
                      st_n=pl.col("strel").is_not_null().sum())
    mk = ["jcd", "motor_no", "mperiod"]
    md = daily_sums(ev.filter(pl.col("motor_no").is_not_null()), tg.filter(pl.col("motor_no").is_not_null()),
                    mk, motor_sums)
    motor = expanding(md, mk, list(motor_sums), "").join(
        window(md, mk, ["n", "rpts", "tj", "tj_n"], "30d", "_30"), on=mk + ["d"])
    motor = motor.select(
        *mk, "d", m_n=pl.col("n"), m_top2=(pl.col("top2") + 10 / 3) / (pl.col("n") + 10),
        m_top3=(pl.col("top3") + 5) / (pl.col("n") + 10), m_rpts=pl.col("rpts") / (pl.col("n") + 10),
        m_rwin=pl.col("rwin") / (pl.col("n") + 10), m_tj=pl.col("tj") / (pl.col("tj_n") + 5),
        m_strel=pl.col("strel") / (pl.col("st_n") + 10),
        m_n_30=pl.col("n_30"), m_rpts_30=pl.col("rpts_30") / (pl.col("n_30") + 5),
        m_tj_30=pl.col("tj_30") / (pl.col("tj_n_30") + 3))
    hk = ["jcd", "boat_no"]
    hd = daily_sums(ev.filter(pl.col("boat_no").is_not_null()), tg.filter(pl.col("boat_no").is_not_null()), hk,
                    dict(n=pl.len(), rpts=pl.col("rpts").sum(), tj=pl.col("tenjirel").sum(),
                         tj_n=pl.col("tenjirel").is_not_null().sum()))
    hull = window(hd, hk, ["n", "rpts", "tj", "tj_n"], "120d", "").select(
        *hk, "d", h_n=pl.col("n"), h_rpts=pl.col("rpts") / (pl.col("n") + 10), h_tj=pl.col("tj") / (pl.col("tj_n") + 5))

    # ---- 今節（同じ会場での直近7日）----
    sk = ["racer_id", "jcd"]
    sd = daily_sums(ev, tg, sk, dict(
        n=pl.len(), days=pl.col("d").n_unique(), win=pl.col("win").sum(), pts=pl.col("pts").sum(),
        rpts=pl.col("rpts").sum(), tj=pl.col("tenjirel").sum(), tj_n=pl.col("tenjirel").is_not_null().sum(),
        st=pl.col("st_num").sum(), st_n=pl.col("st_num").is_not_null().sum(), strel=pl.col("strel").sum()))
    series = window(sd, sk, ["n", "days", "win", "pts", "rpts", "tj", "tj_n", "st", "st_n", "strel"], "7d", "").select(
        *sk, "d", s_n=pl.col("n"), s_days=pl.col("days"), s_win=pl.col("win"),
        s_pts=pl.col("pts") / pl.col("n"), s_rpts=pl.col("rpts") / (pl.col("n") + 2),
        s_tj=pl.col("tj") / (pl.col("tj_n") + 1), s_st=pl.col("st") / pl.col("st_n"),
        s_strel=pl.col("strel") / (pl.col("st_n") + 2))

    # ---- 展示タイム ----
    if tenji is None:
        tenji = results.select(*BK, "racer_id", tenji=pl.when(pl.col("tenji") > 0).then(pl.col("tenji")))
        tg = tg.join(tenji, on=BK + ["racer_id"], how="left")
    else:
        tg = tg.join(tenji.select(*BK, "tenji"), on=BK, how="left")

    # ---- レース ----
    race = meta.select(*KEY, "min_of_day",
                       race_type=pl.col("race_type").replace_strict(RACE_TYPES, default=0, return_dtype=pl.Int64),
                       fixed_entry=pl.col("race_type").str.contains("進入固定").cast(pl.Int64),
                       series=pl.col("series").replace_strict(SERIES, default=0, return_dtype=pl.Int64),
                       ladies=pl.col("series").is_in(["ヴィーナス", "オールレディース"]))
    female = (entries.join(race.filter(pl.col("ladies")).select(KEY), on=KEY)
              .group_by("racer_id").agg(female_from=pl.col("date").min()))

    df = (tg.join(racer, on=["racer_id", "d"], how="left")
          .join(lane, on=["racer_id", "boat", "d"], how="left")
          .join(vb, on=["jcd", "boat", "d"], how="left")
          .join(motor, on=mk + ["d"], how="left")
          .join(hull, on=hk + ["d"], how="left")
          .join(series, on=sk + ["d"], how="left")
          .join(race.drop("ladies"), on=KEY, how="left")
          .join(female, on="racer_id", how="left"))
    df = df.with_columns(
        # 女子戦に出た日以降は女子と分かる（それより前の行には付けない）
        female=(pl.col("date") >= pl.col("female_from")).fill_null(False).cast(pl.Int64),
        month=pl.col("d").dt.month(),
        tenji_rel=pl.col("tenji") - pl.col("tenji").mean().over(KEY),
        tenji_rank=pl.col("tenji").rank("average").over(KEY),
    ).with_columns(
        tenji_z=pl.col("tenji_rel") / (pl.col("tenji").std().over(KEY) + 0.01),
        tenji_adj_racer=pl.col("tenji_rel") - pl.col("r_tj_all"),
        tenji_adj_motor=pl.col("tenji_rel") - pl.col("m_tj"),
    )
    full = df.group_by(KEY).agg(n=pl.len(), b=pl.col("boat").n_unique()).filter(
        (pl.col("n") == 6) & (pl.col("b") == 6)).select(KEY)
    return df.join(full, on=KEY).drop("d", "female_from").sort(BK)


RACE_LEVEL = ["jcd", "race_no", "min_of_day", "month", "race_type", "fixed_entry", "series"]
ENTRY = ["class_ord", "age", "weight", "nat_wr", "nat_2r", "loc_wr", "loc_2r", "motor_2r", "boat_2r",
         "racer_id", "female"]
TENJI = ["tenji", "tenji_rel", "tenji_rank", "tenji_z", "tenji_adj_racer", "tenji_adj_motor"]


def per_boat_columns(df):
    skip = set(BK + RACE_LEVEL + ["deadline", "name", "branch", "class", "motor_no", "boat_no", "setsu_seiseki",
                                  "mperiod", "rank"])
    return [c for c in df.columns if c not in skip]


def labels(results):
    """正常レース（6艇完走・着順1〜6が1つずつ）の着順。"""
    lab = results.select(*BK, "racer_id", rank=pl.col("rank_raw").str.extract(r"^0([1-6])$", 1).cast(pl.Int8))
    ok = (lab.group_by(KEY).agg(n=pl.len(), r=pl.col("rank").drop_nulls().n_unique())
          .filter((pl.col("n") == 6) & (pl.col("r") == 6)).select(KEY))
    return lab.join(ok, on=KEY)


def main():
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet")
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet")
    meta = pl.read_parquet(ROOT / "data/parsed/race_meta.parquet")
    df = build(entries, results, meta)
    df = df.join(labels(results), on=BK + ["racer_id"], how="left")
    out = ROOT / "data/features/v3.parquet"
    df.write_parquet(out)
    print(f"{out}: {df.height:,} rows, {df.height // 6:,} races, {df['date'].min()}〜{df['date'].max()}, "
          f"{len(per_boat_columns(df))} per-boat columns")


if __name__ == "__main__":
    main()
