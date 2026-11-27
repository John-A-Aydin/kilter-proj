"""Checks for the grading baselines in experiments/: the BERT-style encoder, batching, tree features, Holm."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import graders  # noqa: E402
from grade_climbs import holm  # noqa: E402
from kilter.data import FINISH, FOOT, HAND, START, Board  # noqa: E402
try:  # LightGBM is in the optional "experiments" extra.
    from lightgbm_grader import geometry_features  # noqa: E402
except ImportError:
    geometry_features = None

BOARD = Board.from_csv(ROOT / "data" / "training" / "hold_vocabulary.csv")


def climb(*holds):
    """Token ids for (dense hold, role) pairs."""
    return [Board.token(dense, role) for dense, role in holds]


class EncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = graders.BertConfig("test", d_model=32, n_layers=2, n_heads=4, dropout=0.0)
        self.model = graders.GradeEncoder(config, BOARD).eval()
        self.tokens = torch.tensor([climb((3, START), (40, HAND), (100, HAND), (200, FINISH), (7, FOOT)) + [0, 0],
                                    climb((5, START), (6, START), (300, FINISH), (9, FOOT), (11, FOOT), (60, HAND), (70, HAND))])
        self.angle = torch.tensor([40 / 90, 25 / 90])

    def test_hold_order_does_not_matter(self):
        reordered = self.tokens.clone()
        reordered[0, :5] = self.tokens[0, [4, 2, 0, 3, 1]]
        reordered[1] = self.tokens[1].flip(0)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.tokens, self.angle), self.model(reordered, self.angle))

    def test_padding_does_not_matter(self):
        padded = torch.cat([self.tokens, torch.zeros(2, 6, dtype=torch.long)], dim=1)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.tokens, self.angle), self.model(padded, self.angle))

    def test_angle_changes_the_grade(self):
        with torch.no_grad():
            self.assertFalse(torch.allclose(self.model(self.tokens, self.angle), self.model(self.tokens, self.angle * 0)))


class BatchTest(unittest.TestCase):
    def test_every_training_row_once_per_epoch_with_little_padding(self):
        rng = np.random.default_rng(0)
        lengths = torch.from_numpy(rng.integers(3, 30, 5000))
        tokens = torch.zeros(5000, 30, dtype=torch.long)
        for i, n in enumerate(lengths):
            tokens[i, :n] = 4
        data = {"tokens": tokens, "lengths": lengths}
        rows = torch.arange(0, 5000, 2)
        batches = list(graders.batches(data, rows, 16, torch.Generator().manual_seed(0)))
        seen = torch.cat([index for _, index in batches])
        self.assertEqual(sorted(seen.tolist()), rows.tolist())
        self.assertEqual(len(batches), graders.steps_per_epoch(len(rows), 16))
        for batch, index in batches:
            self.assertEqual(batch.shape[1], int(lengths[index].max()))
        padding = 1 - sum(int(lengths[i].sum()) for _, i in batches) / sum(b.numel() for b, _ in batches)
        self.assertLess(padding, 0.1)


@unittest.skipIf(geometry_features is None, "needs the experiments extra (lightgbm)")
class TreeFeatureTest(unittest.TestCase):
    def test_counts_spans_and_gaps(self):
        holds = climb((3, START), (40, HAND), (100, HAND), (200, FINISH), (7, FOOT))
        features, names = geometry_features(np.asarray([holds + [0]]), BOARD)
        value = dict(zip(names, features[0]))
        self.assertEqual((value["holds"], value["starts"], value["hands"], value["finishes"], value["feet"]), (5, 1, 2, 1, 1))
        ys = np.sort(BOARD.y[[3, 40, 100, 200]].astype(np.float64))
        self.assertAlmostEqual(value["hand_y_span"], ys[-1] - ys[0], places=6)
        self.assertAlmostEqual(value["hand_gap_vertical_max"], np.diff(ys).max(), places=6)
        self.assertAlmostEqual(value["hand_gap_vertical_mean"], np.diff(ys).mean(), places=6)
        self.assertAlmostEqual(value["foot_y_max"], float(BOARD.y[7]), places=6)

    def test_climb_without_feet_gives_missing_values(self):
        features, names = geometry_features(np.asarray([climb((3, START), (200, FINISH))]), BOARD)
        value = dict(zip(names, features[0]))
        self.assertTrue(np.isnan(value["foot_y_max"]))
        self.assertTrue(np.isnan(value["hand_gap_vertical_second"]))


class HolmTest(unittest.TestCase):
    def test_known_values(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
