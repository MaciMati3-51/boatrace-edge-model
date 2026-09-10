import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_preregistered import evaluate, load_tickets


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tickets = [dict(combo='1-2', odds=2., p=.5, q=.5, R1=True, R2=False, R3=True)]
        self.races = {(1, 11): self.tickets, (1, 12): self.tickets}
        self.pays = {k: [dict(combo='1=2', amount=200)] for k in self.races}

    def test_frozen_six(self):
        races = load_tickets()
        self.assertEqual(len(races), 6)
        self.assertEqual(sum(t['R2'] for ts in races.values() for t in ts), 0)
        self.assertTrue(all(sum(t['R3'] for t in ts) == 2 for ts in races.values()))

    def test_payout_return_and_no_bets(self):
        report = evaluate(self.races, self.pays)
        self.assertIn('2/2 = 100.0%', report)
        self.assertIn('差=+0.0000', report)
        self.assertIn('| R1 | 2 | 2 | 200.0%', report)
        self.assertIn('算出不可（購入なし）', report)

    def test_missing_result_stops_all_metrics(self):
        del self.pays[(1, 12)]
        report = evaluate(self.races, self.pays)
        self.assertIn('pending', report)
        self.assertNotIn('P2:', report)
        self.assertNotIn('P1:', report)

    def test_odds_movement_fails_gate(self):
        self.pays[(1, 12)][0]['amount'] = 190
        report = evaluate(self.races, self.pays)
        self.assertIn('P1 FAIL', report)
        self.assertNotIn('P2:', report)
        self.assertNotIn('| P3 |', report)

    def test_dead_heat_not_silently_dropped(self):
        self.pays[(1, 11)].append(dict(combo='1=3', amount=200))
        self.assertIn('手動確認', evaluate(self.races, self.pays))

    def test_losing_ticket_counts_in_stake(self):
        tickets = self.tickets + [dict(combo='1-3', odds=5., p=.2, q=.2, R1=True, R2=False, R3=True)]
        races = {k: tickets for k in self.races}
        self.assertIn('| R1 | 4 | 2 | 100.0%', evaluate(races, self.pays))


if __name__ == '__main__':
    unittest.main()
