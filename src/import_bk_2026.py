"""Import requested dates from local official TXT, preserving all other dates."""
from pathlib import Path
import polars as pl
from parse_kb import parse_k_file,parse_b_file
ROOT=Path(__file__).resolve().parents[1]
for kind in ('B','K'):
 collected={'entries':[]} if kind=='B' else {'results':[],'payouts':[]}
 dates=[]
 for path in sorted((ROOT/f'data/raw/txt/{kind}').glob('*/*.TXT')):
  date='20'+path.stem[1:]
  if not '20260101'<=date<='20260909': continue
  if kind=='B': collected['entries'].extend(parse_b_file(path,date))
  else:
   r,p=parse_k_file(path,date); collected['results'].extend(r); collected['payouts'].extend(p)
  dates.append(date)
 for name,rows in collected.items():
  path=ROOT/f'data/parsed/{name}.parquet'; old=pl.read_parquet(path)
  new=pl.DataFrame(rows).select([pl.col(c).cast(t) for c,t in old.schema.items()])
  merged=pl.concat([old.filter(~pl.col('date').is_in(dates)),new])
  tmp=path.with_suffix('.tmp.parquet'); merged.write_parquet(tmp); tmp.replace(path)
  print(name,'imported dates',len(dates),'rows',new.height,flush=True)
