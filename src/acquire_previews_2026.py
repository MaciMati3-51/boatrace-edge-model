"""Download requested 2026-01-01..09-09 daily preview CSVs, newest first.
Retain raw source and provenance. Stop on throttling; no official page crawl.
"""
import csv,io,json,time,hashlib
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
import requests
root=Path(__file__).resolve().parents[1]
out=root/'data/raw/boatracecsv'; out.mkdir(parents=True,exist_ok=True)
s=requests.Session(); d=date(2026,9,9); count=0
with (out/'acquisition.jsonl').open('a') as log:
 while d>=date(2026,1,1):
  for kind in ('tkz','stt','sui','od2'):
   path=out/kind/(d.strftime('%Y%m%d')+'.csv'); url=f'https://boatracecsv.github.io/data/previews/{kind}/{d:%Y/%m/%d}.csv'
   if path.exists(): continue
   r=s.get(url,timeout=30)
   if r.status_code in (429,503): raise RuntimeError(f'Stop on throttling: {r.status_code}, Retry-After={r.headers.get("Retry-After")}')
   status=r.status_code
   if status==200:
    rows=list(csv.DictReader(io.StringIO(r.content.decode('utf-8-sig'))))
    if not rows or 'レースコード' not in rows[0] or any(x['レースコード'][:8]!=d.strftime('%Y%m%d') for x in rows): raise ValueError(url)
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_bytes(r.content); tmp.replace(path)
   elif status!=404: r.raise_for_status()
   log.write(json.dumps(dict(url=url,status=status,rows=len(rows) if status==200 else 0,sha256=hashlib.sha256(r.content).hexdigest(),fetched_at=datetime.now(timezone.utc).isoformat()),ensure_ascii=False)+'\n'); log.flush()
   count+=1; time.sleep(.1)
  if d.day in (1,10,20): print(d,'requests',count,flush=True)
  d-=timedelta(days=1)
print('COMPLETE',count,flush=True)
