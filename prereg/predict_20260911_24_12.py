import sys
sys.path.insert(0, '/home/zeros/L_dev/47_ギャンブル/src')
import predict_day as p
import polars as pl
from datetime import datetime, timezone
rows = [
[1,3783,50,52.2,6.84,39.52,6.55,34.09,35.59,31.48],
[2,4688,33,52.0,6.99,56.41,5.48,37.04,41.27,42.65],
[3,4352,40,52.0,7.45,53.85,7.64,63.51,31.75,45.16],
[4,4704,38,52.0,6.42,46.92,6.06,34.38,44.44,33.93],
[5,4524,38,52.0,6.69,36.62,6.52,34.78,40.00,32.35],
[6,5088,26,44.5,6.59,44.19,6.50,33.33,34.29,38.10],
]
def target(date):
    entries=pl.DataFrame(rows,schema=['boat','racer_id','age','weight','nat_wr','nat_2r','loc_wr','loc_2r','motor_2r','boat_2r'],orient='row').with_columns(date=pl.lit(date),jcd=pl.lit('24'),race_no=pl.lit(12),class_ord=pl.lit(4))
    results=pl.read_parquet(p.ROOT/'data/parsed/results.parquet')
    entries=entries.with_columns(pl.col('jcd').cast(results.schema['jcd']),pl.col('boat').cast(results.schema['boat']),pl.col('racer_id').cast(results.schema['racer_id']))
    racer,vb=p.history_as_of(results,date)
    df=entries.join(racer,on='racer_id',how='left').join(vb,on=['jcd','boat'],how='left').sort(p.KEY+['boat'])
    df.write_csv(p.ROOT/'prereg/input_20260911_24_12.csv')
    return df
p.build_target=target
print('Started:',datetime.now(timezone.utc).isoformat(),flush=True)
sys.argv=['predict_day','--date','20260911','--out',str(p.ROOT/'prereg/pred_20260911_24_12.parquet')]
p.main()
print('Completed:',datetime.now(timezone.utc).isoformat(),flush=True)
df=pl.read_parquet(p.ROOT/'prereg/pred_20260911_24_12.parquet')
print(df.select('first','p_1st').unique().sort('first'))
print(df.sort('p_2t',descending=True).head(6))
print(df.filter(pl.col('p_2f').is_not_null()).sort('p_2f',descending=True).head(6))
