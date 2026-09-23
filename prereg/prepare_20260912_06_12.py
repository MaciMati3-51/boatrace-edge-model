"""Manual transcription of three user screenshots; no race results accessed."""
import sys
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import live_market

entries=[
 [1,5066,4,28,52.3,6.80,54.70,4.12,18.18,50.48,31.05,6.56,0.0,.09],
 [2,4066,3,46,53.3,6.15,41.90,6.06,41.67,31.78,29.28,6.62,0.0,.09],
 [3,4068,2,46,52.3,4.74,22.12,5.07,29.91,42.98,35.38,6.62,-.5,-.02],
 [4,4654,4,40,53.7,6.49,45.16,7.18,60.71,33.63,38.67,6.76,0.0,.05],
 [5,5091,2,25,52.0,6.05,45.45,6.13,43.75,39.81,31.16,6.66,0.0,.17],
 [6,4269,3,40,53.3,7.25,60.71,0.00,0.00,44.64,42.47,6.63,0.0,.12],
]
schema=['boat','racer_id','class_ord','age','weight','nat_wr','nat_2r','loc_wr','loc_2r',
        'motor_2r','boat_2r','tenji','tilt','exhibition_st']
deadline='2026-09-12T17:08:00+09:00'
capture='2026-09-12T16:58:45+09:00'
df=pl.DataFrame(entries,schema=schema,orient='row').with_columns(
    date=pl.lit('20260912'),jcd=pl.lit(6),race_no=pl.lit(12),
    exhibition_weight=pl.col('weight'),adjustment_weight=pl.lit(0.0),exhibition_course=pl.col('boat'),
    wind_speed=pl.lit(3.0),wind_direction=pl.lit(4.0),wave_height=pl.lit(1.0),
    weather_code=pl.lit(2.0),air_temperature=pl.lit(25.0),water_temperature=pl.lit(26.0),
    tkz_captured_at=pl.lit(capture),stt_captured_at=pl.lit(capture),sui_captured_at=pl.lit(capture),
    deadline_at=pl.lit(deadline))
exacta={
 1:[2.4,11.0,3.5,8.4,12.1],
 2:[28.6,100.4,50.2,109.5,150.6],
 3:[25.6,92.7,100.4,172.1,120.5],
 4:[30.9,70.8,92.7,92.7,54.7],
 5:[41.5,200.8,133.9,120.5,86.0],
 6:[36.5,172.1,241.0,86.0,172.1],
}
quinella={
 1:[5.4,6.5,4.7,11.1,4.7],
 2:[27.0,15.7,37.8,31.5],
 3:[12.6,37.8,21.0],
 4:[11.8,17.1],
 5:[37.8],
}
records=[]
for pool,boards in [('2t',exacta),('2f',quinella)]:
 for a,values in boards.items():
  bs=[b for b in range(1,7) if b!=a] if pool=='2t' else list(range(a+1,7))
  assert len(bs)==len(values)
  for b,odd in zip(bs,values):
   records.append(dict(date='20260912',jcd=6,race_no=12,pool=pool,first=a,second=b,odds=odd,
       captured_at='2026-09-12T16:58:40+09:00',deadline_at=deadline))
input_path=ROOT/'prereg/input_20260912_06_12.csv'
odds_path=ROOT/'prereg/odds_20260912_06_12.csv'
output=ROOT/'prereg/live_20260912_06_12'
if input_path.exists() or odds_path.exists() or output.exists():
 raise SystemExit('Refusing to overwrite recorded inputs or prediction')
live_market.live.validate_input(df)
odds=live_market.validate_odds(pl.DataFrame(records))
df.write_csv(input_path)
odds.write_csv(odds_path)
live_market.predict(input_path,odds_path,output)
sources={}
for stamp in ['165830','165840','165845']:
 path=ROOT/f'picture/スクリーンショット 2026-09-12 {stamp}.png'
 sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
metadata=dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_images_sha256=sources,
    race_result_accessed=False,model_retrained=False,scheduled_deadline_jst=deadline,
    odds_display_updated_jst='2026-09-12T16:57+09:00',odds_update_precision='minute',
    weather_observed_at='11R時点',weather_code=2,wind_direction_code=4,
    weather_crosscheck_url='https://boatracecsv.github.io/data/previews/sui/2026/09/12.csv',
    weather_crosscheck_row={'レースコード':'202609120612','取得日時':'2026-09-12T16:58:17.319903+09:00',
       '気象観測時刻':'1608','風速(m)':'3.0','風向':'4','波の高さ(cm)':'1.0','天候':'2','気温(℃)':'25.0','水温(℃)':'26.0'},
    weather_code_reference='https://github.com/BoatraceCSV/fun-site/blob/main/packages/shared/src/utils/weather-pt.ts',
    notes='Screenshots transcribed manually. F.02=-0.02. No subjective adjustments. Histories available locally only; no same-day results fetched.')
(output/'source_record.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print('Saved:',output)
final=pl.read_csv(output/'final.csv')
for pool in ['2t','2f']:
 print(pool,'probability leaders')
 print(final.filter(pl.col('pool')==pool).sort('p_final',descending=True).select('first','second','odds','p_fundamental','p_final','ev','paper_candidate').head(6))
 print('paper candidates')
 print(final.filter((pl.col('pool')==pool)&pl.col('paper_candidate')).sort('ev',descending=True).select('first','second','odds','p_final','ev'))
print('First probabilities:')
print(pl.read_csv(output/'fundamental.csv').select('first','p_1st').unique().sort('first'))
