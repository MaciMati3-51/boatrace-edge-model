"""9/10前向きテストをローカル払戻で評価。取得・学習・封印開封は行わない。

実行: .venv/bin/python src/evaluate_preregistered.py
結果が未収録なら pending。既存レポートには実行ごとに追記する。
"""
import argparse
import csv
import hashlib
import itertools
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from holdout import HOLDOUT_END

ROOT = Path(__file__).resolve().parent.parent
DATE = '20260910'
assert DATE > HOLDOUT_END, f'{DATE} は封印期間内。前向きテストはHOLDOUT_END({HOLDOUT_END})より後の日付のみ'
HASH = '5fa72e5281419bca92363732eec585abc0867eac7e6d6cf51262e52a9f5bc0a9'
PAIRS = {f'{a}-{b}' for a, b in itertools.combinations(range(1, 7), 2)}


def load_tickets():
    frozen = ROOT / f'prereg/pred_{DATE}.parquet'
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != HASH:
        raise ValueError('凍結予測のハッシュが事前登録と不一致')
    predictions = {
        (r['jcd'], r['race_no'], f"{r['first']}-{r['second']}"): r['p_2f']
        for r in pl.read_parquet(frozen).to_dicts() if r['first'] < r['second']
    }
    with (ROOT / f'prereg/{DATE}_screenshot_tickets.csv').open() as f:
        raw = list(csv.DictReader(f))
    races = {}
    for row in raw:
        if row['bet_type'] != '2f':
            continue
        if row['date'] != DATE:
            raise ValueError('対象日以外のデータ')
        key = (int(row['jcd']), int(row['race_no']))
        item = dict(combo=row['combo'], odds=float(row['odds']), p=float(row['p_model']))
        if not all(math.isfinite(item[c]) and item[c] > 0 for c in ('odds', 'p')):
            raise ValueError('非正または非有限のオッズ・確率')
        if not math.isclose(item['p'], predictions[(*key, item['combo'])], abs_tol=1e-12):
            raise ValueError('凍結確率との不一致')
        races.setdefault(key, []).append(item)
    if not races:
        raise ValueError('対象なし')
    for tickets in races.values():
        if len(tickets) != 15 or {t['combo'] for t in tickets} != PAIRS:
            raise ValueError('2連複15目の欠損・重複')
        if not math.isclose(sum(t['p'] for t in tickets), 1, abs_tol=1e-6):
            raise ValueError('確率和が1でない')
        total = sum(1 / t['odds'] for t in tickets)
        for t in tickets:
            t['q'] = 1 / t['odds'] / total
            t['combined'] = t['p'] ** .064 * t['q'] ** 1.017
        normalizer = sum(t['combined'] for t in tickets)
        popular = sorted(tickets, key=lambda t: (t['odds'], t['combo']))
        # 人気2/3の境界に同率がある場合、事前登録には処理規定がない。
        if popular[0]['odds'] == popular[1]['odds'] or popular[2]['odds'] == popular[3]['odds']:
            raise ValueError('人気境界の同率: R3の処理が事前登録に未定義')
        for t in tickets:
            t['combined'] /= normalizer
            t['R1'] = t['p'] * t['odds'] > 1
            t['R2'] = t['combined'] * t['odds'] > 1
            t['R3'] = t in popular[1:3]
    return races


def evaluate(races, payouts, draws=10000):
    """全対象の通常払戻が揃うまで集計しない。特殊払戻は要確認にする。"""
    settled = []
    issues = []
    for key, tickets in sorted(races.items()):
        pay = payouts.get(key, [])
        if not pay:
            issues.append(f'{key[0]:02d}-{key[1]:02d}R: 払戻未収録')
            continue
        if len(pay) != 1:
            issues.append(f'{key}: 複数的中・同着等は手動確認が必要')
            continue
        combo = '-'.join(sorted(pay[0]['combo'].replace('=', '-').split('-')))
        amount = pay[0]['amount']
        if combo not in PAIRS or amount is None or not math.isfinite(amount) or amount <= 0:
            issues.append(f'{key}: 特払い・不成立等は手動確認が必要')
            continue
        hit = next(t for t in tickets if t['combo'] == combo)
        settled.append((key, tickets, hit, amount))
    if issues:
        return '状態: pending / 要確認\n\n' + '\n'.join(issues) + '\n\n全対象が揃うまでP1〜P3は集計しない。'
    lines = ['| 会場-R | 的中2連複 | 撮影オッズ | 払戻/100円 | 一致 |', '|---|---|---:|---:|---|']
    matches = []
    for key, tickets, hit, amount in settled:
        ok = abs(hit['odds'] * 100 - amount) <= 1
        matches.append(ok)
        lines.append(f"| {key[0]:02d}-{key[1]:02d} | {hit['combo']} | {hit['odds']:g} | {amount:g} | {ok} |")
    rate = sum(matches) / len(matches)
    lines += ['', f'P1: {sum(matches)}/{len(matches)} = {rate:.1%}（基準 ≥90%）']
    if rate < .9:
        return '\n'.join(lines + ['', 'P1 FAIL。事前登録どおりP2・P3は無効として集計を停止。'])
    rng = np.random.default_rng(0)
    index = rng.integers(0, len(settled), size=(draws, len(settled)))
    lp = np.array([-math.log(s[2]['p']) for s in settled])
    lq = np.array([-math.log(s[2]['q']) for s in settled])
    delta = lp - lq
    ci = np.quantile(delta[index].mean(axis=1), [.025, .975])
    lines += ['', f'P2: model={lp.mean():.4f}, market={lq.mean():.4f}, 差={delta.mean():+.4f}, 95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]', '', '| P3 | 点数 | 的中数 | 回収率 | 95%CI |', '|---|---:|---:|---:|---|']
    for rule in ('R1', 'R2', 'R3'):
        stakes = np.array([sum(t[rule] for t in s[1]) for s in settled])
        returns = np.array([s[3] / 100 if s[2][rule] else 0 for s in settled])
        hits = sum(s[2][rule] for s in settled)
        if not stakes.sum():
            lines.append(f'| {rule} | 0 | 0 | 算出不可（購入なし） | — |')
            continue
        denominator = stakes[index].sum(axis=1)
        valid = denominator > 0
        boot = returns[index].sum(axis=1)[valid] / denominator[valid]
        low, high = np.quantile(boot, [.025, .975])
        lines.append(f'| {rule} | {stakes.sum()} | {hits} | {returns.sum()/stakes.sum():.1%} | [{low:.1%}, {high:.1%}] |')
    lines += ['', 'CIはレース単位のペア・ブートストラップ10,000回、seed=0、百分位法。ROIは各点均等賭け、払戻は実払戻を使用。購入ゼロの再標本はROIのCIから除外。小標本のCIは不安定で、優位性を結論しない。']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--payouts', type=Path, default=ROOT/'data/parsed/payouts.parquet')
    ap.add_argument('--output', type=Path, default=ROOT/f'prereg/{DATE}_result.md')
    args = ap.parse_args()
    races = load_tickets()
    # 読み込むのは前向き対象日の2連複のみ。封印期間は評価しない。
    pay = pl.scan_parquet(args.payouts).filter((pl.col('date') == DATE) & (pl.col('bet_type') == '２連複')).collect()
    payouts = {}
    for row in pay.to_dicts():
        key = (row['jcd'], row['race_no'])
        if key in races:
            payouts.setdefault(key, []).append(row)
    result = evaluate(races, payouts)
    stamp = datetime.now(UTC).isoformat()
    preamble = f'\n## 評価実行 {stamp}\n\n対象: スクショ6レース。若松11Rは撮影ファイル名が締切予定後。他5枚は締切予定約8〜45分前。候補生成時刻と前日予測凍結時刻は別。\n\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as f:
        f.write(preamble + result + '\n')
    print(result)


if __name__ == '__main__':
    main()
