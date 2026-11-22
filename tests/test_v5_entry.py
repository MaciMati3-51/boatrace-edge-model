import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_v4 as f4  # noqa: E402
import v5_entry as v5  # noqa: E402


def random_v4(n, seed=0):
    """Plackett-Luce で作った整合的な (p1, joint, tri)。"""
    rng = np.random.default_rng(seed)
    w = rng.gamma(2.0, size=(n, 6))
    p1 = w / w.sum(axis=1, keepdims=True)
    joint = np.stack([p1[:, a] * w[:, b] / (w.sum(axis=1) - w[:, a]) for a, b in f4.PAIRS], axis=1)
    tri = np.stack([p1[:, a] * w[:, b] / (w.sum(axis=1) - w[:, a])
                    * w[:, c] / (w.sum(axis=1) - w[:, a] - w[:, b]) for a, b, c in f4.TRIPLES], axis=1)
    return p1, joint, tri


class EntryCorrectionTest(unittest.TestCase):
    def test_identity_when_entry_is_by_lane(self):
        p1, joint, tri = random_v4(5)
        course = np.tile(v5.BOATS, (5, 1))
        st = np.full((5, 6), np.nan)
        coef = dict(use_st=True, beta=[np.ones(len(v5.names(True, 1))), np.ones(len(v5.names(True, 2))),
                                       np.ones(len(v5.names(True, 3)))])
        q1, qj, qt = v5.correct(p1, joint, tri, course, st, coef)
        assert np.allclose(q1, p1) and np.allclose(qj, joint) and np.allclose(qt, tri)


    def test_corrected_distributions_are_consistent(self):
        p1, joint, tri = random_v4(20, seed=1)
        rng = np.random.default_rng(2)
        course = np.array([rng.permutation(6) + 1 for _ in range(20)])
        st = rng.uniform(-0.05, 0.3, (20, 6))
        coef = dict(use_st=True, beta=[rng.normal(size=len(v5.names(True, k))) for k in (1, 2, 3)])
        q1, qj, qt = v5.correct(p1, joint, tri, course, st, coef)
        assert np.allclose(q1.sum(axis=1), 1) and np.allclose(qj.sum(axis=1), 1) and np.allclose(qt.sum(axis=1), 1)
        # 周辺が上の段と一致する
        jm = np.zeros((20, 6, 6))
        jm[:, f4.PAIRS[:, 0], f4.PAIRS[:, 1]] = qj
        assert np.allclose(jm.sum(axis=2), q1)
        tm = np.zeros((20, 6, 6, 6))
        tm[:, f4.TRIPLES[:, 0], f4.TRIPLES[:, 1], f4.TRIPLES[:, 2]] = qt
        assert np.allclose(tm.sum(axis=3), jm)


    def test_fit_recovers_course_effect(self):
        """展示で外から内へ入った艇が強くなる世界を作り、1着段の係数の符号を回収できること。"""
        n = 4000
        p1, joint, tri = random_v4(n, seed=3)
        rng = np.random.default_rng(4)
        course = np.tile(v5.BOATS, (n, 1))
        for i in range(0, n, 2):  # 半分のレースで 5号艇が2コースに入る
            course[i] = [1, 3, 4, 5, 2, 6]
        st = np.full((n, 6), np.nan)
        true = dict(use_st=False, beta=[np.r_[0, 0, 0, 0, 0, 0.8, 0, 0, 0], np.zeros(11), np.zeros(11)])
        q1, qj, qt = v5.correct(p1, joint, tri, course, st, true)
        order = np.zeros((n, 6), int)
        for i in range(n):
            tm = qt[i] / qt[i].sum()
            order[i, :3] = f4.TRIPLES[rng.choice(len(tm), p=tm)]
            order[i, 3:] = [b for b in range(6) if b not in order[i, :3]]
        coef = v5.fit(p1, joint, tri, order, course, st, use_st=False)
        moved_in = coef["names"][0].index("moved_in")
        assert coef["beta"][0][moved_in] > 0.3


    def test_arrays_rejects_non_permutation(self):
        keys = pl.DataFrame({"date": ["20260101", "20260101"], "jcd": [1, 1], "race_no": [1, 2]})
        ex = pl.DataFrame({"date": ["20260101", "20260101"], "jcd": [1, 1], "race_no": [1, 2],
                           **{f"c{b}": [b, 1 if b == 2 else b] for b in v5.BOATS},
                           **{f"s{b}": [0.1, 0.1] for b in v5.BOATS}})
        course, st = v5.arrays(keys, ex)
        assert (course[1] == v5.BOATS).all() and np.isnan(st[1]).all()
        assert np.isfinite(st[0]).all()


if __name__ == "__main__":
    unittest.main()


class CourseDeltaTest(unittest.TestCase):
    def synthetic(self):
        rng = np.random.default_rng(5)
        rows = []
        for day in range(20):
            date = f"202601{day + 1:02d}"
            for race in range(1, 4):
                racers = rng.permutation(12)[:6] + 100
                courses = rng.permutation(6) + 1 if race == 3 else np.arange(1, 7)
                for boat, (rid, c) in enumerate(zip(racers, courses), start=1):
                    rows.append(dict(date=date, jcd=1, race_no=race, boat=boat, racer_id=int(rid),
                                     entry_course=int(c), rank_raw=f"0{rng.integers(1, 7)}"))
        return pl.DataFrame(rows)

    def test_causal_and_zero_by_lane(self):
        import v5b_course as vb
        self.addCleanup(setattr, vb, "PRIOR_END", vb.PRIOR_END)
        vb.PRIOR_END = "20260110"
        res = self.synthetic()
        target = res.filter(pl.col("date") == "20260115").sort(["race_no", "boat"])
        keys = target.select(v5.KEY).unique(maintain_order=True)
        course = np.array([[2, 1, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
        full = vb.deltas(target.select(*v5.KEY, "boat", "racer_id"), course,
                         res.select("date", "racer_id", "entry_course", "rank_raw"))
        past = vb.deltas(target.select(*v5.KEY, "boat", "racer_id"), course,
                         res.filter(pl.col("date") < "20260115").select("date", "racer_id", "entry_course",
                                                                        "rank_raw"))
        self.assertTrue(np.allclose(full, past))
        self.assertTrue((full[1] == 0).all() and (full[0, 2:] == 0).all())
        self.assertTrue(np.abs(full[2]).sum() > 0)
        self.assertEqual(len(keys), 3)
