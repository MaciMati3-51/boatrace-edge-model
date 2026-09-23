import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import live_market as market


def board(pool='2f'):
    return pl.DataFrame([dict(date='20260912',jcd=18,race_no=4,pool=pool,
        first=a,second=b,odds=float(5+i),captured_at='2026-09-12T12:12:00+09:00',
        deadline_at='2026-09-12T12:17:00+09:00') for i,(a,b) in enumerate(market.POOLS[pool])])


class MarketTest(unittest.TestCase):
    def test_validate_separate_pools(self):
        market.validate_odds(board('2t'))
        market.validate_odds(board('2f'))
        market.validate_odds(pl.concat([board('2t'),board('2f')]))
        for bad in [board().head(14),pl.concat([board(),board().head(1)]),
                    board().with_columns(odds=pl.lit(0.)),
                    board().with_columns(odds=pl.lit(float('inf'))),
                    board().with_columns(captured_at=pl.lit('2026-09-12T12:17:00+09:00'))]:
            with self.assertRaises(ValueError): market.validate_odds(bad)

    def test_combination_limits_and_normalization(self):
        p=np.array([[.7,.2,.1],[.1,.2,.7]])
        q=np.array([[.2,.4,.4],[.4,.4,.2]])
        np.testing.assert_allclose(market.distribution(p,q,[0,1]),q)
        np.testing.assert_allclose(market.distribution(p,q,[1,0]),p)
        np.testing.assert_allclose(market.distribution(p,q,[.5,2]).sum(axis=1),1)

    def test_calibration_can_ignore_useless_fundamental(self):
        p=np.tile([1/3]*3,(100,1))
        q=np.tile([.7,.2,.1],(100,1))
        y=np.repeat([0,1,2],[70,20,10])
        c=market.fit(p,q,y)
        self.assertAlmostEqual(c[1],1,places=3)
        self.assertLessEqual(-np.log(market.distribution(p,q,c)[np.arange(100),y]).mean(),
                             -np.log(p[np.arange(100),y]).mean())

    def test_full_saved_pipeline(self):
        if not (market.OUT/'market_coefficients.json').exists():
            self.skipTest('Train market artifact first')
        live=market.live
        source=pl.read_csv(live.OUT/'input_example.csv',schema_overrides={'date':pl.String})
        source=source.with_columns(date=pl.lit('20260912'),**{
            c:pl.col(c).str.replace(r'^\d{4}-\d{2}-\d{2}','2026-09-12')
            for c in live.TIMES+['deadline_at']})
        row=source.row(0,named=True)
        odds=pl.concat([board('2t'),board('2f')]).with_columns(
            jcd=pl.lit(row['jcd']),race_no=pl.lit(row['race_no']),
            deadline_at=pl.lit(row['deadline_at']),captured_at=pl.lit(row['tkz_captured_at']))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source.write_csv(root/'input.csv'); odds.write_csv(root/'odds.csv')
            market.predict(root/'input.csv',root/'odds.csv',root/'output')
            result=pl.read_csv(root/'output/final.csv')
            self.assertEqual(result.height,45)
            for pool in market.POOLS:
                self.assertAlmostEqual(result.filter(pl.col('pool')==pool)['p_final'].sum(),1)
            np.testing.assert_allclose(result['ev'].to_numpy(),result['p_final'].to_numpy()*result['odds'].to_numpy())
            with self.assertRaisesRegex(ValueError,'already exists'):
                market.predict(root/'input.csv',root/'odds.csv',root/'output')


if __name__=='__main__': unittest.main()
