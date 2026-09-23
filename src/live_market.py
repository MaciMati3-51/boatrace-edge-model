"""Market calibration per pool, retrospective evaluation, and live CSV pipeline.

train uses only out-of-sample fundamental predictions; odds never enter live_model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp

import live_model as live

KEY=live.KEY
OUT=live.OUT
SPLIT='20260901'
POOLS={'2t':[(a,b) for a in range(1,7) for b in range(1,7) if a!=b],
       '2f':[(a,b) for a in range(1,7) for b in range(a+1,7)]}
LABELS={'2t':'２連単','2f':'２連複'}


def distribution(p,q,coef):
    z=coef[0]*np.log(np.clip(p,1e-15,1))+coef[1]*np.log(np.clip(q,1e-15,1))
    return np.exp(z-logsumexp(z,axis=-1,keepdims=True))


def fit(p,q,y,market_only=False):
    lp,lq=np.log(np.clip(p,1e-15,1)),np.log(np.clip(q,1e-15,1))
    features=np.stack([lq] if market_only else [lp,lq],axis=-1)
    r=np.arange(len(y))
    def objective(coef):
        z=features@coef
        logprob=z-logsumexp(z,axis=1,keepdims=True)
        residual=np.exp(logprob)
        residual[r,y]-=1
        return -logprob[r,y].mean(),np.einsum('ij,ijk->k',residual,features)/len(y)
    result=minimize(objective,[1.] if market_only else [.1,1.],jac=True,
                    method='L-BFGS-B',bounds=[(0,5)]*(1 if market_only else 2))
    if not result.success:
        raise ValueError(f'Calibration failed: {result.message}')
    return [0.,float(result.x[0])] if market_only else result.x.tolist()


def load_pool(pool):
    probs=pl.read_parquet(OUT/'preview_evaluation.parquet')
    odds=pl.read_parquet(live.ROOT/'data/parsed/boatracecsv/od2.parquet')
    odds=odds.filter((pl.col('minutes_before_scheduled_deadline')>0)&(pl.col('minutes_before_scheduled_deadline')<=60))
    if odds.unique(KEY).height!=odds.height:
        raise ValueError('Duplicate market records')
    boards={tuple(row[c] for c in KEY):row for row in odds.iter_rows(named=True)}
    labels=pl.read_parquet(OUT/'dataset.parquet')
    winners={}
    for group in labels.partition_by(KEY):
        key=tuple(group[c][0] for c in KEY)
        first=group.filter(pl.col('rank')==1)['boat'][0]
        second=group.filter(pl.col('rank')==2)['boat'][0]
        winners[key]=(first,second) if pool=='2t' else tuple(sorted((first,second)))
    pay=pl.read_parquet(live.ROOT/'data/parsed/payouts.parquet').filter(
        (pl.col('date')>='20260820')&(pl.col('date')<=live.END)&(pl.col('bet_type')==LABELS[pool]))
    payouts={}
    for row in pay.iter_rows(named=True):
        key=tuple(row[c] for c in KEY)
        payouts.setdefault(key,[]).append(row)
    records=[]
    pairs=POOLS[pool]
    col='p_2t' if pool=='2t' else 'p_2f'
    source_prefix='2連単_' if pool=='2t' else '2連複_'
    sep='-' if pool=='2t' else '='
    counters={'prediction_races':0,'missing_or_invalid_board':0,'invalid_payout':0}
    for group in probs.partition_by(KEY):
        counters['prediction_races']+=1
        key=tuple(group[c][0] for c in KEY)
        board=boards.get(key)
        try:
            o=np.array([float(board[f'{source_prefix}{a}{sep}{b}']) for a,b in pairs])
            if not np.isfinite(o).all() or (o<1).any():
                raise ValueError('Invalid odds')
        except (KeyError,TypeError,ValueError):
            counters['missing_or_invalid_board']+=1
            continue
        pp={(row['first'],row['second']):row[col] for row in group.iter_rows(named=True)}
        p=np.array([pp[pair] for pair in pairs],dtype=float)
        if not np.isfinite(p).all() or not np.isclose(p.sum(),1):
            raise ValueError('Invalid model probabilities')
        pays=payouts.get(key,[])
        winner=winners[key]
        if len(pays)!=1 or pays[0]['combo']!=f'{winner[0]}-{winner[1]}' or pays[0]['amount']<=0:
            counters['invalid_payout']+=1
            continue
        q=1/o; q/=q.sum()
        records.append((key,p,q,o,pairs.index(winner),pays[0]['amount']))
    if not records:
        raise ValueError(f'No matched races: {pool}')
    return dict(keys=[r[0] for r in records],p=np.stack([r[1] for r in records]),
                q=np.stack([r[2] for r in records]),odds=np.stack([r[3] for r in records]),
                y=np.array([r[4] for r in records]),amount=np.array([r[5] for r in records])),counters


def day_ci(values,dates):
    unique=np.unique(dates)
    sums=np.array([values[dates==d].sum() for d in unique])
    counts=np.array([(dates==d).sum() for d in unique])
    picks=np.random.default_rng(0).integers(0,len(unique),(2000,len(unique)))
    samples=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()


def roi_ci(stakes,returns,dates):
    unique=np.unique(dates)
    s=np.array([stakes[dates==d].sum() for d in unique])
    r=np.array([returns[dates==d].sum() for d in unique])
    picks=np.random.default_rng(0).integers(0,len(unique),(2000,len(unique)))
    den=s[picks].sum(axis=1); num=r[picks].sum(axis=1)
    valid=den>0
    return np.quantile(num[valid]/den[valid],[.025,.975]).tolist() if valid.any() else None


def train():
    report={'calibration_period':['20260820','20260831'],'evaluation_period':[SPLIT,live.END],
            'status':'exploratory historical comparison; normal six-finisher races only',
            'bet_rule':'all combinations with p * captured odds > 1; 100 JPY each',
            'pools':{}}
    saved={}
    examples=[]
    evaluation_records=[]
    for pool in POOLS:
        d,counters=load_pool(pool)
        dates=np.array([k[0] for k in d['keys']])
        tr=dates<SPLIT; te=~tr
        if tr.sum()<50 or te.sum()<50:
            raise ValueError('Insufficient calibration/test races')
        power=fit(d['p'][tr],d['q'][tr],d['y'][tr],True)
        combined=fit(d['p'][tr],d['q'][tr],d['y'][tr])
        predictions={'fundamental':d['p'][te],'market':d['q'][te],
                     'market_power':distribution(d['p'][te],d['q'][te],power),
                     'combined':distribution(d['p'][te],d['q'][te],combined)}
        result={'counts':counters,'calibration_races':int(tr.sum()),'test_races':int(te.sum()),
                'market_power_coefficients':power,'combined_coefficients':combined,'metrics':{}}
        y=d['y'][te]; r=np.arange(len(y)); losses={}
        for name,p in predictions.items():
            loss=-np.log(np.clip(p[r,y],1e-15,1)); losses[name]=loss
            bets=p*d['odds'][te]>1
            stakes=bets.sum(axis=1)*100
            returns=bets[r,y]*d['amount'][te]
            result['metrics'][name]={'logloss':float(loss.mean()),'bets':int(bets.sum()),
                                     'stake_jpy':int(stakes.sum()),'return_jpy':int(returns.sum()),
                                     'roi':float(returns.sum()/stakes.sum()) if stakes.sum() else None,
                                     'day_bootstrap_roi_95ci':roi_ci(stakes,returns,dates[te])}
            testkeys=[key for key,keep in zip(d['keys'],te) if keep]
            for i,key in enumerate(testkeys):
                for j,(a,b) in enumerate(POOLS[pool]):
                    evaluation_records.append(dict(date=key[0],jcd=key[1],race_no=key[2],pool=pool,
                        method=name,first=a,second=b,probability=float(p[i,j]),
                        odds=float(d['odds'][te][i,j]),paper_selected=bool(bets[i,j]),
                        realized_return_jpy=int(d['amount'][te][i]) if j==y[i] else 0))
        diff=losses['combined']-losses['market_power']
        result['combined_minus_market_power']={'mean':float(diff.mean()),'day_bootstrap_95ci':day_ci(diff,dates[te])}
        report['pools'][pool]=result
        saved[pool]={'combined':fit(d['p'],d['q'],d['y']),
                     'market_power':fit(d['p'],d['q'],d['y'],True),
                     'calibration_races':len(y)+int(tr.sum())}
        # Export the captured odds input format using one historical race.
        key=d['keys'][-1]
        source=pl.read_parquet(live.ROOT/'data/parsed/boatracecsv/od2.parquet').filter(
            (pl.col('date')==key[0])&(pl.col('jcd')==key[1])&(pl.col('race_no')==key[2])).row(0,named=True)
        for (a,b),odd in zip(POOLS[pool],d['odds'][-1]):
            examples.append(dict(date=key[0],jcd=key[1],race_no=key[2],pool=pool,first=a,second=b,
                                 odds=float(odd),captured_at=source['取得日時'],
                                 deadline_at=f'{key[0][:4]}-{key[0][4:6]}-{key[0][6:]}T{source["締切時刻"]}:00+09:00'))
        print(pool,json.dumps(result,ensure_ascii=False),flush=True)
    saved.update(training_through=live.END,generated_at=datetime.now(timezone.utc).isoformat(),
                 fundamental_model_sha256=hashlib.sha256((OUT/'preview_model.pkl').read_bytes()).hexdigest(),
                 note='Final coefficients fit on Aug20-Sep09 OOS base probabilities; base subsequently refit through Sep09')
    (OUT/'market_coefficients.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
    pl.DataFrame(examples).write_csv(OUT/'odds_input_example.csv')
    pl.DataFrame(evaluation_records).write_parquet(OUT/'market_evaluation.parquet')
    (live.ROOT/'prereg/live_market_v1_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


def validate_odds(df):
    expected=KEY+['pool','first','second','odds','captured_at','deadline_at']
    if set(expected)-set(df.columns) or not df.height:
        raise ValueError('Missing odds input columns or empty input')
    if set(df['pool'].to_list())-set(POOLS):
        raise ValueError('pool must be 2t or 2f')
    for group in df.partition_by(KEY+['pool']):
        pool=group['pool'][0]
        pairs=list(zip(group['first'],group['second']))
        if sorted(pairs)!=POOLS[pool]:
            raise ValueError(f'Incomplete or duplicate {pool} board')
        a=group['odds'].to_numpy().astype(float)
        if not np.isfinite(a).all() or (a<1).any():
            raise ValueError('Odds must all be finite and >= 1')
        if group['captured_at'].n_unique()!=1 or group['deadline_at'].n_unique()!=1:
            raise ValueError('Mixed timestamps within odds board')
        deadline=datetime.fromisoformat(group['deadline_at'][0])
        stamp=datetime.fromisoformat(group['captured_at'][0])
        if deadline.utcoffset() is None or stamp.utcoffset() is None:
            raise ValueError('Timezone required')
        if deadline.strftime('%Y%m%d')!=group['date'][0] or not 0<(deadline-stamp).total_seconds()<=3600:
            raise ValueError('Odds outside pre-deadline window')
    return df


def predict(input_path,odds_path,out):
    inputs=live.validate_input(pl.read_csv(input_path,schema_overrides={'date':pl.String}))
    odds=validate_odds(pl.read_csv(odds_path,schema_overrides={'date':pl.String}))
    inputkeys=inputs.select(KEY+['deadline_at']).unique()
    if odds.select(KEY+['deadline_at']).unique().join(inputkeys,on=KEY+['deadline_at'],how='anti').height:
        raise ValueError('Odds race/deadline does not match exhibition input')
    if inputkeys.select(KEY).join(odds.select(KEY).unique(),on=KEY,how='anti').height:
        raise ValueError('Missing odds for an input race')
    coef=json.loads((OUT/'market_coefficients.json').read_text())
    sha=hashlib.sha256((OUT/'preview_model.pkl').read_bytes()).hexdigest()
    if sha!=coef['fundamental_model_sha256']:
        raise ValueError('Fundamental model changed; recalibrate market coefficients')
    if out.exists():
        raise ValueError('Output directory already exists')
    out.mkdir(parents=True)
    live.predict(input_path,out/'fundamental.csv')
    p=pl.read_csv(out/'fundamental.csv',schema_overrides={'date':pl.String})
    records=[]
    for group in odds.partition_by(KEY+['pool']):
        group=group.sort('first','second'); row=group.row(0,named=True); pool=row['pool']
        pp=p.filter(pl.all_horizontal([pl.col(c)==row[c] for c in KEY]))
        if pool=='2f': pp=pp.filter(pl.col('first')<pl.col('second'))
        pp=pp.sort('first','second')['p_2t' if pool=='2t' else 'p_2f'].to_numpy()
        o=group['odds'].to_numpy(); q=1/o; q/=q.sum()
        final=distribution(pp,q,coef[pool]['combined'])
        for original,raw,market,prob in zip(group.iter_rows(named=True),pp,q,final):
            records.append(dict(**original,p_fundamental=float(raw),p_market=float(market),
                                p_final=float(prob),ev=float(prob*original['odds']),
                                paper_candidate=bool(prob*original['odds']>1)))
    pl.DataFrame(records).write_csv(out/'final.csv')
    (out/'metadata.json').write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),
        model_sha256=sha,calibration=coef,odds_sha256=hashlib.sha256(odds_path.read_bytes()).hexdigest(),
        note='Research probabilities and paper candidates; profitability not established. Check timestamp before claiming preregistration.'),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['train','predict'])
    ap.add_argument('--input',type=Path); ap.add_argument('--odds',type=Path); ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.command=='train': train()
    elif args.input and args.odds and args.out: predict(args.input,args.odds,args.out)
    else: ap.error('predict requires --input, --odds and --out')
