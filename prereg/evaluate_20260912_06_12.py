"""Compare frozen live forecast to the user-provided result screenshot."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'prereg/live_20260912_06_12'
if (OUT/'result.json').exists():
    raise SystemExit('Result already recorded; refusing overwrite')
frozen=json.loads((OUT/'freeze_hashes.json').read_text())
for name,sha in frozen.items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha, f'Frozen file changed: {name}'
actual=json.loads((OUT/'result_input.json').read_text())
final=pl.read_csv(OUT/'final.csv')
assert final.height==45 and final.filter(pl.col('paper_candidate')).height==4
assert not actual['refunds'] and sorted(actual['finish_order'])==list(range(1,7))
rows=[]; scores={}; totals={}
for pool in ['2t','2f']:
    board=final.filter(pl.col('pool')==pool).sort('p_final',descending=True)
    win=board.filter((pl.col('first')==1)&(pl.col('second')==2)).row(0,named=True)
    payout=actual['payout_per_100_jpy'][pool]['1-2']
    scores[pool]={'winning_combo':'1-2','probability_rank':1,'p_fundamental':win['p_fundamental'],
        'p_market':win['p_market'],'p_final':win['p_final'],
        'fundamental_logloss':-math.log(win['p_fundamental']),
        'market_logloss':-math.log(win['p_market']),'final_logloss':-math.log(win['p_final']),
        'captured_odds':win['odds'],'settled_multiplier':payout/100,
        'odds_change_fraction':payout/100/win['odds']-1,
        'captured_ev':win['ev'],'final_probability_times_settled_multiplier':win['p_final']*payout/100}
    for row in board.filter(pl.col('paper_candidate')).iter_rows(named=True):
        combo=f'{row["first"]}-{row["second"]}'
        paid=actual['payout_per_100_jpy'][pool].get(combo,0)
        rows.append(dict(pool=pool,combo=combo,stake_jpy=100,return_jpy=paid,
                         captured_odds=row['odds'],p_final=row['p_final'],captured_ev=row['ev']))
    selected=[r for r in rows if r['pool']==pool]
    stake=sum(r['stake_jpy'] for r in selected); paid=sum(r['return_jpy'] for r in selected)
    totals[pool]=dict(stake_jpy=stake,return_jpy=paid,profit_jpy=paid-stake,roi=paid/stake)
stake=sum(r['stake_jpy'] for r in rows); paid=sum(r['return_jpy'] for r in rows)
totals['combined']=dict(stake_jpy=stake,return_jpy=paid,profit_jpy=paid-stake,roi=paid/stake)
report=dict(evaluated_at=datetime.now(timezone.utc).isoformat(),frozen_hashes_verified=True,
    actual=actual,result_image_sha256=hashlib.sha256((ROOT/actual['source_image']).read_bytes()).hexdigest(),
    scores=scores,paper_tickets=rows,totals=totals,
    notes='No reforecast or retraining. Stakes follow frozen rule. Settled-odds EV is diagnostic only, not a retrospective selection rule.')
(OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
pl.DataFrame(rows).write_csv(OUT/'result_tickets.csv')
print(json.dumps(report,ensure_ascii=False,indent=2))
