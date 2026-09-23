import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_model as live


def fixture():
    rows = []
    for b in range(1, 7):
        row = dict.fromkeys(live.ENTRY + live.RAW, 1.0)
        row.update(date="20260912", jcd=18, race_no=4, boat=b, racer_id=4000+b,
                   exhibition_course=b, tenji=6.9, exhibition_st=.01*b,
                   deadline_at="2026-09-12T12:17:00+09:00")
        for col in live.TIMES:
            row[col] = "2026-09-12T12:12:00+09:00"
        for col in live.BASE:
            row.setdefault(col, .5)
        rows.append(row)
    return pl.DataFrame(rows)


class LiveModelTest(unittest.TestCase):
    def test_reject_missing_boat_late_and_backfilled_data(self):
        df = fixture()
        live.validate_input(df)
        for bad in [df.head(5),
                    df.with_columns(tkz_captured_at=pl.lit("2026-09-12T12:17:00+09:00")),
                    df.with_columns(stt_captured_at=pl.lit("backfill-from-daily")),
                    df.with_columns(exhibition_course=pl.lit(1)),
                    df.with_columns(tenji=pl.lit(float('nan')))]:
            with self.assertRaises((ValueError, TypeError)):
                live.validate_input(bad)

    def test_baseline_matrices_match_existing_code(self):
        df=fixture().sort(live.BK)
        m=live.RaceModel(live.BASE)
        x=m.matrix(df)
        np.testing.assert_equal(m.first_matrix(x),live.baseline.stage1_matrix(df))
        np.testing.assert_equal(m.pair_matrix(x,np.array([0]),np.array([1])),
                                live.baseline.pair_matrix(x,np.array([0]),np.array([1])))

    def test_ties_flying_and_no_result_features(self):
        df=fixture().with_columns(exhibition_st=pl.when(pl.col('boat')==6).then(-.03).otherwise(pl.col('exhibition_st')))
        df=live.derive(live.validate_input(df))
        self.assertEqual(df['tenji_rank'].to_list(),[3.5]*6)
        self.assertEqual(df['exhibition_f'].to_list(),[0,0,0,0,0,1])
        m=live.RaceModel(live.BASE+live.EXTRA)
        x=m.matrix(df)
        modified=df.with_columns(rank=pl.lit(1),st=pl.lit(.01),entry_course=pl.lit(6))
        np.testing.assert_equal(x,m.matrix(modified))

    def test_probability_marginals(self):
        df=fixture()
        p1=np.full((1,6),1/6)
        joint=np.full((1,30),1/30)
        out=live.probability_rows(df,p1,joint)
        self.assertEqual(out.height,30)
        self.assertAlmostEqual(out['p_2t'].sum(),1)
        self.assertAlmostEqual(out['p_2f'].sum(),1)
        for b in range(1,7):
            self.assertAlmostEqual(out.filter(pl.col('first')==b)['p_2t'].sum(),1/6)

    def test_history_does_not_include_same_day(self):
        from build_features import racer_history
        data=pl.DataFrame(dict(racer_id=[1,1,1],date=['20260910','20260911','20260911'],
                               rank_raw=['02','01','01'],st_num=[.2,.1,.1]))
        row=racer_history(data).filter(pl.col('date')=='20260911').row(0,named=True)
        self.assertEqual(row['hist_n'],1)
        self.assertEqual(row['hist_win_rate'],0)
        self.assertEqual(row['hist_mean_st'],.2)

    def test_saved_artifact_and_prediction_entrypoint(self):
        artifact=live.OUT/'preview_model.pkl'
        if not artifact.exists():
            self.skipTest('Run training to create local artifact')
        model=live.load_model(artifact)
        df=pl.read_parquet(live.OUT/'dataset.parquet').sort(live.BK).head(6)
        p1,joint=model.predict(df)
        self.assertAlmostEqual(p1.sum(),1)
        self.assertAlmostEqual(joint.sum(),1)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            copy=directory/'model.pkl'
            live.save_model(model,copy)
            actual=live.load_model(copy).predict(df)
            np.testing.assert_equal(joint,actual[1])
            original=pl.read_csv(live.OUT/'input_example.csv',schema_overrides={'date':pl.String})
            path=directory/'synthetic_input.csv'
            target=directory/'synthetic_prediction.csv'
            original.write_csv(path)
            with self.assertRaisesRegex(ValueError,'training cutoff'):
                live.predict(path,target)
            # Synthetic date shift exercises the interface, not a real prediction.
            original=original.with_columns(date=pl.lit('20260912'),**{
                c:pl.col(c).str.replace(r'^\d{4}-\d{2}-\d{2}','2026-09-12')
                for c in live.TIMES+['deadline_at']})
            original.write_csv(path)
            live.predict(path,target)
            output=pl.read_csv(target)
            self.assertEqual(output.height,30)
            self.assertAlmostEqual(output['p_2t'].sum(),1)
            self.assertAlmostEqual(output['p_2f'].sum(),1)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                live.predict(path,target)


if __name__ == '__main__':
    unittest.main()
