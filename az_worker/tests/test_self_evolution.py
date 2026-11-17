from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from az_worker import arena, train
from az_worker.board import BLACK, WHITE, Board
from az_worker.evolve import advance_state, decide_promotion, should_promote
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
    def test_the_opening_is_legal_and_repeatable(self) -> None:
        first = arena.random_opening(random.Random(1), 4)
        second = arena.random_opening(random.Random(1), 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        board = Board()
        for index, (row, col) in enumerate(first):
            self.assertEqual(board.grid[row][col], 0)
            board.play(row, col, BLACK if index % 2 == 0 else WHITE)

    def test_an_az_series_counts_every_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            save(small_model(), str(first))
            save(small_model(), str(second))
            candidate = arena.az_side(first, "cpu", 8, 0.05, 8, seed=1)
            opponent = arena.az_side(second, "cpu", 8, 0.05, 8, seed=2)
            result = arena.play_series(
                candidate,
                opponent,
                games=2,
                opening_moves=2,
                seed=7,
            )
            self.assertEqual(result.games, 2)
            self.assertGreaterEqual(result.winrate, 0.0)
            self.assertLessEqual(result.winrate, 1.0)

    def test_the_promotion_threshold_counts_a_draw_as_half(self) -> None:
        self.assertTrue(should_promote(arena.MatchResult(5, 4, 1), 0.55))
        self.assertFalse(should_promote(arena.MatchResult(5, 5, 0), 0.55))
        self.assertTrue(should_promote(arena.MatchResult(50, 50, 0), 0.5))
        self.assertFalse(should_promote(arena.MatchResult(0, 0, 0), 0.0))


class ProgressionTest(unittest.TestCase):
    def test_the_learner_advances_even_when_the_candidate_is_rejected(self) -> None:
        state = {
            "generation": 1,
            "champion": "champion.pt",
            "learner": "champion.pt",
            "baseline": "baseline.pt",
            "history": [],
        }
        advance_state(
            state,
            2,
            Path("champion.pt"),
            Path("candidate.pt"),
            False,
            arena.MatchResult(3, 5, 2),
            arena.MatchResult(6, 4, 0),
        )
        self.assertEqual(state["champion"], "champion.pt")
        self.assertEqual(state["learner"], "candidate.pt")
        self.assertEqual(state["generation"], 2)
        self.assertFalse(state["history"][0]["promoted"])
        self.assertEqual(state["history"][0]["teacher"], "champion.pt")
        self.assertEqual(state["history"][0]["baseline_winrate"], 0.6)

    def test_a_promoted_candidate_becomes_the_champion(self) -> None:
        state = {
            "generation": 1,
            "champion": "old.pt",
            "learner": "old.pt",
            "history": [],
        }
        advance_state(
            state,
            2,
            Path("old.pt"),
            Path("new.pt"),
            True,
            arena.MatchResult(6, 4, 0),
            None,
        )
        self.assertEqual(state["champion"], "new.pt")
        self.assertEqual(state["learner"], "new.pt")
        self.assertNotIn("baseline_winrate", state["history"][0])

    def test_beating_the_champion_is_not_enough_when_the_baseline_collapses(self) -> None:
        gate = arena.MatchResult(15, 5, 0)
        self.assertTrue(should_promote(gate, 0.55))
        self.assertFalse(
            decide_promotion(gate, 0.55, arena.MatchResult(3, 17, 0), 0.45)
        )
        self.assertTrue(
            decide_promotion(gate, 0.55, arena.MatchResult(12, 8, 0), 0.45)
        )
        self.assertTrue(decide_promotion(gate, 0.55, None, 0.45))
        self.assertFalse(
            decide_promotion(
                arena.MatchResult(5, 5, 0),
                0.55,
                arena.MatchResult(20, 0, 0),
                0.45,
            )
        )


class SelfPlayGeneratorTest(unittest.TestCase):
    def test_training_does_not_touch_the_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = root / "generator.pt"
            seed_model = root / "seed.pt"
            output = root / "output.pt"
            save(small_model(), str(generator))
            save(small_model(), str(seed_model))
            before = generator.read_bytes()
            code = train.main(
                [
                    "--init",
                    str(seed_model),
                    "--model",
                    str(output),
                    "--selfplay-model",
                    str(generator),
                    "--teacher-data",
                    "",
                    "--games",
                    "1",
                    "--iterations",
                    "1",
                    "--simulations",
                    "8",
                    "--train-steps",
                    "2",
                    "--batch-size",
                    "8",
                    "--buffer",
                    "64",
                    "--device",
                    "cpu",
                    "--seed",
                    "5",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(generator.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
