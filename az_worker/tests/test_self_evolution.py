from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from az_worker import arena
from az_worker.evolve import should_promote
from az_worker.features import PLANE_COUNT
from az_worker.network import GomokuNet, save
from az_worker.teacher import VALUE_OUTCOME, VALUE_SEARCH, collect_az_game

DEVICE = torch.device("cpu")


def small_model() -> GomokuNet:
    return GomokuNet(channels=8, blocks=1, value_hidden=8)


class AzTeacherTest(unittest.TestCase):
    def test_the_search_teacher_gives_a_normalised_policy(self) -> None:
        examples = collect_az_game(
            small_model(),
            DEVICE,
            simulations=16,
            time_budget=None,
            candidate_width=8,
            value_target=VALUE_SEARCH,
            seed=3,
        )
        self.assertGreater(len(examples), 5)
        for example in examples:
            self.assertEqual(example.planes.shape, (PLANE_COUNT, 15, 15))
            self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=4)
            self.assertTrue(np.all(example.policy[example.legal == 0] == 0))
            self.assertGreaterEqual(example.teacher_cell, 0)
            self.assertLessEqual(abs(example.value), 1.0)

    def test_the_outcome_targets_are_final_scores(self) -> None:
        examples = collect_az_game(
            small_model(),
            DEVICE,
            simulations=16,
            time_budget=None,
            candidate_width=8,
            value_target=VALUE_OUTCOME,
            seed=4,
        )
        self.assertGreater(len(examples), 5)
        self.assertTrue({example.value for example in examples} <= {-1.0, 0.0, 1.0})

    def test_an_unknown_value_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collect_az_game(small_model(), DEVICE, 8, None, 8, "eval")


class ArenaTest(unittest.TestCase):
    def test_an_az_series_counts_every_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            save(small_model(), str(first))
            save(small_model(), str(second))
            candidate = arena.az_side(first, "cpu", 8, 0.05, 8, seed=1)
            opponent = arena.az_side(second, "cpu", 8, 0.05, 8, seed=2)
            result = arena.play_series(candidate, opponent, games=2)
            self.assertEqual(result.games, 2)
            self.assertGreaterEqual(result.winrate, 0.0)
            self.assertLessEqual(result.winrate, 1.0)

    def test_the_promotion_threshold_counts_a_draw_as_half(self) -> None:
        self.assertTrue(should_promote(arena.MatchResult(5, 4, 1), 0.55))
        self.assertFalse(should_promote(arena.MatchResult(5, 5, 0), 0.55))
        self.assertTrue(should_promote(arena.MatchResult(50, 50, 0), 0.5))
        self.assertFalse(should_promote(arena.MatchResult(0, 0, 0), 0.0))


if __name__ == "__main__":
    unittest.main()
