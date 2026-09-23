"""Two-stage probabilities with verified pre-deadline exhibition inputs.

Only local data is used. Backfilled previews and the sealed period are excluded.
Run: OMP_NUM_THREADS=4 .venv/bin/python src/live_model.py train
Predict: ... live_model.py predict --input race.csv --out prediction.csv
The CSV input schema is saved as data/models/live_v1/input_example.csv.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import pickle
from itertools import groupby
from pathlib import Path

import numpy as np
import polars as pl
import sklearn

from build_features import racer_history, venue_boat_history
from holdout import HOLDOUT_END
import predict_day as baseline

ROOT = Path(__file__).resolve().parents[1]
KEY = baseline.KEY
BK = KEY + ["boat"]
OUT = ROOT / "data/models/live_v1"
BASE = baseline.FEATURES.copy()
RAW = ["exhibition_weight", "adjustment_weight", "tenji", "tilt",
       "exhibition_course", "exhibition_st", "wind_speed", "wind_direction",
       "wave_height", "weather_code", "air_temperature", "water_temperature"]
EXTRA = RAW + ["tenji_rel", "tenji_rank", "exhibition_st_rel", "exhibition_f",
               "course_change"]
TIMES = ["tkz_captured_at", "stt_captured_at", "sui_captured_at"]
ENTRY = ["racer_id", "class_ord", "age", "weight", "nat_wr", "nat_2r",
         "loc_wr", "loc_2r", "motor_2r", "boat_2r"]
TRAIN_START, SPLIT, END = "20260719", "20260820", "20260909"


def validate_input(df):
    """Validate screenshot transcription before deriving features or inference."""
    needed = BK + ENTRY + RAW + TIMES + ["deadline_at"]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if not df.height or df.unique(BK).height != df.height:
        raise ValueError("Empty input or duplicate race/boat")
    for _, rows in groupby(df.sort(BK).iter_rows(named=True),key=lambda r:tuple(r[c] for c in KEY)):
        validate_records(list(rows))
    values = df.select(RAW).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite preview values")
    if not df.filter(pl.col("tenji").is_between(5, 9)).height == df.height:
        raise ValueError("Exhibition time outside expected bounds")
    if not df.filter(pl.col("exhibition_st").is_between(-1, 1)).height == df.height:
        raise ValueError("Exhibition ST must use signed seconds (F is negative)")
    return df.sort(BK)


def validate_records(rows):
    if sorted(r['boat'] for r in rows) != list(range(1,7)):
        raise ValueError('Each race must contain boats 1..6 exactly once')
    if sorted(r['exhibition_course'] for r in rows) != list(range(1,7)):
        raise ValueError('Exhibition courses must be a permutation of 1..6')
    for col in TIMES+['deadline_at']:
        if len({r[col] for r in rows}) != 1:
            raise ValueError(f'Inconsistent timestamp: {col}')
    row=rows[0]
    deadline=datetime.fromisoformat(row['deadline_at'])
    if deadline.utcoffset() is None or deadline.strftime('%Y%m%d') != row['date']:
        raise ValueError('Deadline timezone/date missing or incorrect')
    for col in TIMES:
        stamp=datetime.fromisoformat(row[col])
        if stamp.utcoffset() is None or not 0<(deadline-stamp).total_seconds()<=3600:
            raise ValueError(f'Preview outside pre-deadline window: {col}')
    if not np.isfinite(np.array([[r[c] for c in RAW] for r in rows],dtype=float)).all():
        raise ValueError('Missing/nonfinite previews')
    if any(not 5<=r['tenji']<=9 or not -1<=r['exhibition_st']<=1 for r in rows):
        raise ValueError('Invalid exhibition values')


def derive(df):
    return df.with_columns(
        tenji_rel=pl.col("tenji")-pl.col("tenji").mean().over(KEY),
        tenji_rank=pl.col("tenji").rank("average").over(KEY),
        exhibition_st_rel=pl.col("exhibition_st")-pl.col("exhibition_st").mean().over(KEY),
        exhibition_f=(pl.col("exhibition_st") < 0).cast(pl.Float64),
        course_change=pl.col("exhibition_course")-pl.col("boat"),
    )


def load_previews():
    frames = {}
    for kind in ("tkz", "stt", "sui"):
        frame = pl.read_parquet(ROOT / f"data/parsed/boatracecsv/{kind}.parquet")
        frame = frame.filter((pl.col("date") >= TRAIN_START) & (pl.col("date") <= END)
                             & (pl.col("minutes_before_scheduled_deadline") > 0)
                             & (pl.col("minutes_before_scheduled_deadline") <= 60))
        if frame.unique(KEY).height != frame.height:
            raise ValueError(f"Duplicate source keys: {kind}")
        frames[kind] = frame
    tk = {"exhibition_weight": "体重(kg)", "adjustment_weight": "体重調整(kg)",
          "tenji": "展示タイム", "tilt": "チルト"}
    st = {"exhibition_course": "コース", "exhibition_st": "スタート展示"}
    long = []
    for kind, mapping in (("tkz", tk), ("stt", st)):
        boats = []
        for b in range(1, 7):
            boats.append(frames[kind].select(
                *KEY, pl.lit(b).cast(pl.Int64).alias("boat"),
                pl.col("取得日時").alias(f"{kind}_captured_at"),
                *[pl.col(f"艇{b}_{source}").cast(pl.Float64, strict=False).alias(name)
                  for name, source in mapping.items()],
                *([pl.concat_str([pl.col("date").str.strptime(pl.Date, "%Y%m%d").cast(pl.String),
                                   pl.lit("T"), pl.col("締切時刻"), pl.lit(":00+09:00")]).alias("deadline_at")]
                  if kind == "tkz" else [])))
        long.append(pl.concat(boats))
    weather = {"wind_speed": "風速(m)", "wind_direction": "風向", "wave_height": "波の高さ(cm)",
               "weather_code": "天候", "air_temperature": "気温(℃)", "water_temperature": "水温(℃)"}
    w = frames["sui"].select(*KEY, pl.col("取得日時").alias("sui_captured_at"),
                             *[pl.col(v).cast(pl.Float64, strict=False).alias(k) for k,v in weather.items()])
    return long[0].join(long[1], on=BK, validate="1:1").join(w, on=KEY, validate="m:1")


def build_dataset():
    entries = pl.read_parquet(ROOT / "data/parsed/entries.parquet").filter(
        (pl.col("date") >= TRAIN_START) & (pl.col("date") <= END)).with_columns(
        class_ord=pl.col("class").replace_strict(baseline.CLASS_ORD))
    results = pl.read_parquet(ROOT / "data/parsed/results.parquet").filter(pl.col("date") <= END)
    # These expanding histories are shifted by DAY, so a race cannot see today's results.
    historic = results.with_columns(st_num=pl.col("st").str.extract(r"^(0?\.\d+)$", 1).cast(pl.Float64))
    rh, vb = racer_history(historic), venue_boat_history(historic)
    df = entries.join(load_previews(), on=BK, validate="1:1")
    df = df.join(rh, on=["racer_id", "date"], validate="m:1").join(vb, on=["jcd", "boat", "date"], validate="m:1")
    labels = results.select(*BK, "racer_id", pl.col("rank_raw").str.extract(r"^0([1-6])$",1).cast(pl.Int8).alias("rank"))
    df = df.join(labels, on=BK+["racer_id"], validate="1:1")
    accepted, rejected = [], 0
    for key, records in groupby(df.sort(BK).iter_rows(named=True),key=lambda r:tuple(r[c] for c in KEY)):
        records=list(records)
        try:
            validate_records(records)
            if any(r['rank'] is None for r in records) or sorted(r['rank'] for r in records) != list(range(1,7)):
                raise ValueError("Not six distinct finishers")
            accepted.append(key)
        except (ValueError, TypeError):
            rejected += 1
    if not accepted:
        raise ValueError("No complete verified races")
    df = derive(df.join(pl.DataFrame(accepted,schema=KEY,orient='row'),on=KEY,validate='m:1').sort(BK))
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT / "dataset.parquet")
    df.filter(pl.col("date") == df["date"].max()).head(6).select(
        BK+ENTRY+RAW+TIMES+["deadline_at"]).write_csv(OUT / "input_example.csv")
    return df, rejected


class RaceModel:
    def __init__(self, features):
        self.features = features
        self.context = baseline.CONTEXT + [c for c in EXTRA if c in features]

    def matrix(self, df):
        return df.select(self.features).to_numpy().astype(float).reshape(-1,6,len(self.features))

    def first_matrix(self, x):
        ci = [i for i,c in enumerate(self.features) if c not in ("boat", "jcd")]
        return np.column_stack([x[:,0,self.features.index("jcd")], x[:,:,ci].reshape(len(x),-1)])

    def pair_matrix(self, x, first, second):
        row = np.arange(len(first))
        ci = [self.features.index(c) for c in self.context]
        cand, prev = x[row,second], x[row,first]
        return np.column_stack([cand, prev[:,ci], cand[:,ci]-prev[:,ci]])

    def fit(self, df):
        x = self.matrix(df)
        rank = df["rank"].to_numpy().reshape(-1,6)
        first = rank.argmin(axis=1)
        self.first = baseline.model([0]).set_params(early_stopping=True).fit(self.first_matrix(x), first)
        r,b = np.nonzero(rank>=2)
        self.second = baseline.model([0,1]).set_params(early_stopping=True).fit(self.pair_matrix(x[r],first[r],b), rank[r,b]==2)
        self.training_through = df["date"].max()
        return self

    def predict(self, df):
        x = self.matrix(df)
        p1 = self.first.predict_proba(self.first_matrix(x))
        if list(self.first.classes_) != list(range(6)):
            raise ValueError("Model did not learn all six classes")
        pairs = baseline.PAIRS
        r = np.repeat(np.arange(len(x)),30)
        scores = self.second.predict_proba(self.pair_matrix(x[r],np.tile(pairs[:,0],len(x)),np.tile(pairs[:,1],len(x))))[:,1].reshape(-1,6,5)
        conditional = scores / scores.sum(axis=2,keepdims=True)
        joint = (p1[:,:,None]*conditional).reshape(-1,30)
        if not np.isfinite(joint).all() or not np.allclose(joint.sum(axis=1),1):
            raise ValueError("Invalid probabilities")
        return p1, joint


def probability_rows(df, p1, joint):
    keys = df.select(KEY).unique(maintain_order=True)
    pairs = baseline.PAIRS
    lookup = {tuple(pair):i for i,pair in enumerate(pairs)}
    rows=[]
    for i,key in enumerate(keys.iter_rows(named=True)):
        for j,(a,b) in enumerate(pairs):
            rows.append(dict(**key,first=int(a+1),second=int(b+1),p_1st=float(p1[i,a]),
                             p_2t=float(joint[i,j]),p_2f=float(joint[i,j]+joint[i,lookup[(b,a)]]) if a<b else None))
    return pl.DataFrame(rows)


def losses(df, p1, joint):
    order = np.argsort(df["rank"].to_numpy().reshape(-1,6),axis=1)
    ix = {tuple(pair):i for i,pair in enumerate(baseline.PAIRS)}
    j = np.array([ix[tuple(v[:2])] for v in order])
    reverse = np.array([ix[(v[1],v[0])] for v in order])
    r = np.arange(len(order))
    return np.column_stack([-np.log(np.clip(p1[r,order[:,0]],1e-15,1)),
                            -np.log(np.clip(joint[r,j],1e-15,1)),
                            -np.log(np.clip(joint[r,j]+joint[r,reverse],1e-15,1))])


def save_model(model, path):
    payload=pickle.dumps(dict(features=model.features,first=model.first,second=model.second,
                              training_through=model.training_through))
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path):
    # Only load locally generated trusted artifacts; pickle is executable data.
    with path.open('rb') as f:
        saved=pickle.load(f)
    model=RaceModel(saved['features'])
    model.first,model.second=saved['first'],saved['second']
    model.training_through=saved['training_through']
    return model


def train():
    if TRAIN_START <= HOLDOUT_END:
        raise ValueError("Development must not overlap sealed holdout")
    df, rejected = build_dataset()
    tr, te = df.filter(pl.col("date") < SPLIT), df.filter(pl.col("date") >= SPLIT)
    print(f"Verified races: train={tr.height//6}, test={te.height//6}, rejected={rejected}",flush=True)
    previous=ROOT/'prereg/live_v1_comparison.json'
    diagnosis=ROOT/'prereg/live_v1_initial_no_early_stopping.json'
    if previous.exists() and not diagnosis.exists():
        diagnosis.write_bytes(previous.read_bytes())
    report = dict(training=[TRAIN_START,SPLIT],evaluation=[SPLIT,END],
                  training_end_inclusive=tr['date'].max(),train_races=tr.height//6,test_races=te.height//6,
                  rejected_races=rejected,features=BASE+EXTRA,backfilled_data_used=False,sealed_evaluation_used=False,
                  early_stopping=True,comparison_status='exploratory: early stopping enabled after diagnosing small-sample overfitting')
    results={}
    for name,features in (("baseline",BASE),("preview",BASE+EXTRA)):
        print(f"Fitting {name}",flush=True)
        model=RaceModel(features).fit(tr)
        p1,joint=model.predict(te)
        results[name]=losses(te,p1,joint)
        probability_rows(te,p1,joint).write_parquet(OUT/f'{name}_evaluation.parquet')
        report[name]=dict(zip(['first_logloss','exacta_logloss','quinella_logloss'],results[name].mean(axis=0).tolist()))
        report[name]['iterations']=[int(model.first.n_iter_),int(model.second.n_iter_)]
        print(name,report[name],flush=True)
    # Paired bootstrap by DAY, retaining the race weighting in each sample.
    difference=results['preview']-results['baseline']
    dates=te.select(KEY).unique(maintain_order=True)['date'].to_numpy()
    unique=np.unique(dates)
    sums=np.array([difference[dates==d].sum(axis=0) for d in unique])
    counts=np.array([(dates==d).sum() for d in unique])
    picks=np.random.default_rng(0).integers(0,len(unique),(2000,len(unique)))
    boots=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)[:,None]
    report['preview_minus_baseline']={'mean':difference.mean(axis=0).tolist(),
        'day_bootstrap_95ci':np.quantile(boots,[.025,.975],axis=0).T.tolist(),
        'order':['1着','2連単','2連複'],'negative_is_better':True}
    report['limitation']='Single historical split; equal training samples; not evidence of profitability or comparison with the larger historical production fit.'
    print('Fitting deployable preview model through '+END,flush=True)
    model=RaceModel(BASE+EXTRA).fit(df)
    report['model_sha256']=save_model(model,OUT/'preview_model.pkl')
    report['dataset_sha256']=hashlib.sha256((OUT/'dataset.parquet').read_bytes()).hexdigest()
    report['versions']={'sklearn':sklearn.__version__,'polars':pl.__version__,'numpy':np.__version__}
    report['production_training_through']=model.training_through
    report['recorded_at']=datetime.now(timezone.utc).isoformat()
    (OUT/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'prereg/live_v1_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def predict(path,out):
    df=validate_input(pl.read_csv(path,schema_overrides={'date':pl.String}))
    model=load_model(OUT/'preview_model.pkl')
    if df['date'].min() <= model.training_through:
        raise ValueError("Prediction date must be after model training cutoff")
    enriched=[]
    results=pl.read_parquet(ROOT/'data/parsed/results.parquet')
    for date in df['date'].unique():
        rh,vb=baseline.history_as_of(results,date)
        enriched.append(df.filter(pl.col('date')==date).join(rh,on='racer_id',validate='m:1')
                        .join(vb,on=['jcd','boat'],validate='m:1'))
    tgt=derive(pl.concat(enriched).sort(BK))
    if tgt.height != df.height:
        raise ValueError("Missing racer/venue history")
    p1,joint=model.predict(tgt)
    if out.exists():
        raise ValueError("Refusing to overwrite prediction")
    out.parent.mkdir(parents=True,exist_ok=True)
    probability_rows(tgt,p1,joint).write_csv(out)
    out.with_suffix('.json').write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(),training_through=model.training_through,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((OUT/'preview_model.pkl').read_bytes()).hexdigest(),
        note='Check generation time against deadline; odds not used in fundamental model.'),indent=2)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['train','predict'])
    ap.add_argument('--input',type=Path)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.command=='train': train()
    elif args.input and args.out: predict(args.input,args.out)
    else: ap.error('predict requires --input and --out')
