"""Keep external CSVs separate and report key coverage; no model evaluation."""
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta
import polars as pl
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/parsed/boatracecsv'; OUT.mkdir(exist_ok=True)
keys=['date','jcd','race_no']
sets={}; report={}
for kind in ('tkz','stt','sui','od2'):
 frames=[]
 for path in sorted((ROOT/'data/raw/boatracecsv'/kind).glob('*.csv')):
  df=pl.read_csv(path,infer_schema=False).with_columns(source_file=pl.lit(str(path.relative_to(ROOT))))
  frames.append(df)
 if not frames: continue
 df=pl.concat(frames,how='diagonal').with_columns(
  date=pl.col('レースコード').str.slice(0,8),
  jcd=pl.col('レースコード').str.slice(8,2).cast(pl.Int64),
  race_no=pl.col('レースコード').str.slice(10,2).cast(pl.Int64))
 if df.select(keys).unique().height!=df.height: raise ValueError(f'Duplicate keys {kind}')
 df.write_parquet(OUT/f'{kind}.parquet')
 sets[kind]=df.select(keys)
 report[kind]={'files':len(frames),'races':df.height,'start':df['date'].min(),'end':df['date'].max()}
 for c in ('取得日時','締切時刻'):
  if c not in df.columns: raise ValueError(f'Missing provenance {c}')
 # Independently check acquisition time against the scheduled deadline.
 minutes=[]
 for race_date,deadline,captured in df.select('date','締切時刻','取得日時').iter_rows():
  try:
   limit=datetime.strptime(race_date+' '+deadline,'%Y%m%d %H:%M').replace(tzinfo=timezone(timedelta(hours=9)))
   stamp=datetime.fromisoformat(captured)
   if stamp.tzinfo is None: raise ValueError('timezone missing')
   minutes.append((limit-stamp).total_seconds()/60)
  except (TypeError,ValueError): minutes.append(None)
 report[kind]['timing_unknown']=sum(x is None for x in minutes)
 report[kind]['at_or_after_scheduled_deadline']=sum(x is not None and x<=0 for x in minutes)
 df=df.with_columns(pl.Series('minutes_before_scheduled_deadline',minutes,dtype=pl.Float64))
 df.write_parquet(OUT/f'{kind}.parquet')
 if kind=='od2':
  columns=[c for c in df.columns if c.startswith(('2連単_','2連複_'))]
  if len(columns)!=45: raise ValueError('Expected 45 odds columns')
  valid=pl.all_horizontal([pl.col(c).cast(pl.Float64,strict=False)>0 for c in columns]).fill_null(False)
  report[kind]['positive_complete_boards']=df.filter(valid).height
for n in ('entries','results'):
 d=pl.read_parquet(ROOT/f'data/parsed/{n}.parquet').filter((pl.col('date')>='20260101')&(pl.col('date')<='20260909')).select(keys).unique()
 sets[n]=d; report[n]={'races':d.height}
common=sets['entries']
for n,d in sets.items():
 if n!='entries': common=common.join(d,on=keys,how='inner')
common.sort(keys,descending=True).write_parquet(OUT/'complete_race_keys.parquet')
report['all_sources_and_results_matched']=common.height
report['parts_exchange']='not provided by these CSVs; missing'
report['training_or_evaluation_performed']=False
(ROOT/'prereg/acquisition_20260101_20260909_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
