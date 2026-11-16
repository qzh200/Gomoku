from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np
import torch

from az_worker.board import BLACK, WHITE, Board
from az_worker.config import Settings
from az_worker.engine import AlphaZeroEngine, EngineError
from az_worker.features import (
    PLANE_COUNT,
    PLANE_LAST,
    PLANE_OTHER,
    PLANE_OTHER_FIVE,
    PLANE_OTHER_OPEN_FOUR,
    PLANE_SELF,
    PLANE_SELF_FIVE,
    PLANE_SELF_OPEN_FOUR,
    TRANSFORMS,
    encode,
    legal_mask,
    transform_planes,
    transform_policy,
)
from az_worker.mcts import MCTS, Node, cell_of, row_col_of
from az_worker.network import ACTION_COUNT, GomokuNet, load_model
from az_worker.selfplay import Example, ReplayBuffer, SelfPlay, augment, train

DEVICE = torch.device("cpu")


def board(stones: list[tuple[int, int, int]]) -> list[list[int]]:
    grid = [[0] * 15 for _ in range(15)]
    for row, col, player in stones:
        grid[row][col] = player
    return grid


def small_model() -> GomokuNet:
    return GomokuNet(channels=8, blocks=1, value_hidden=8)


def permute_cell(cell: tuple[int, int], index: int) -> tuple[int, int]:
    forward, _ = TRANSFORMS[index]
    row, col = divmod(forward(cell[0] * 15 + cell[1]), 15)
    return row, col


def permute_board(position: Board, index: int) -> Board:
    forward, _ = TRANSFORMS[index]
    grid = [[0] * 15 for _ in range(15)]
    for row, col, stone in position.stones:
        new_row, new_col = divmod(forward(row * 15 + col), 15)
        grid[new_row][new_col] = stone
    return Board(grid)


class BoardTests(unittest.TestCase):
    def test_play_and_undo_restore_the_board(self) -> None:
        position = Board(board([(7, 7, BLACK)]))
        position.play(7, 8, WHITE)
        self.assertEqual(position.grid[7][8], WHITE)
        position.undo()
        self.assertEqual(position.grid[7][8], 0)
        self.assertEqual(len(position.stones), 1)

    def test_recognises_a_five(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK)]))
        self.assertTrue(position.wins_with(7, 6, BLACK))
        self.assertTrue(position.wins_with(7, 11, BLACK))
        self.assertFalse(position.wins_with(7, 11, WHITE))

    def test_play_reports_a_winning_move(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK)]))
        self.assertTrue(position.play(7, 6, BLACK))

    def test_offers_the_neighbours_of_the_stones(self) -> None:
        position = Board(board([(7, 7, BLACK)]))
        cells = position.candidates(radius=2)
        self.assertIn((7, 8), cells)
        self.assertIn((6, 6), cells)
        self.assertNotIn((7, 7), cells)
        self.assertNotIn((0, 0), cells)


class FeatureTests(unittest.TestCase):
    def test_the_planes_follow_the_side_to_move(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        planes = encode(position, BLACK, (7, 8))
        self.assertEqual(planes.shape, (PLANE_COUNT, 15, 15))
        self.assertEqual(planes[PLANE_SELF][7][7], 1.0)
        self.assertEqual(planes[PLANE_OTHER][7][8], 1.0)
        self.assertEqual(planes[PLANE_LAST][7][8], 1.0)
        swapped = encode(position, WHITE, (7, 8))
        self.assertEqual(swapped[PLANE_SELF][7][8], 1.0)
        self.assertEqual(swapped[PLANE_OTHER][7][7], 1.0)

    def test_the_threat_planes_mark_the_winning_cells(self) -> None:
        position = Board(
            board([(7, 6, BLACK), (7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)])
        )
        planes = encode(position, BLACK, None)
        self.assertEqual(planes[PLANE_SELF_FIVE][7][5], 1.0)
        self.assertEqual(planes[PLANE_SELF_FIVE][7][10], 1.0)
        self.assertEqual(planes[PLANE_SELF_FIVE].sum(), 2.0)
        self.assertEqual(planes[PLANE_OTHER_FIVE].sum(), 0.0)

    def test_the_threat_planes_mark_the_opponent_winning_cells(self) -> None:
        position = Board(
            board([(7, 6, WHITE), (7, 7, WHITE), (7, 8, WHITE), (7, 9, WHITE)])
        )
        planes = encode(position, BLACK, None)
        self.assertEqual(planes[PLANE_OTHER_FIVE][7][5], 1.0)
        self.assertEqual(planes[PLANE_OTHER_FIVE][7][10], 1.0)
        self.assertEqual(planes[PLANE_OTHER_FIVE].sum(), 2.0)
        self.assertEqual(planes[PLANE_SELF_FIVE].sum(), 0.0)

    def test_the_threat_planes_mark_the_open_fours(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)]))
        planes = encode(position, BLACK, None)
        self.assertEqual(planes[PLANE_SELF_OPEN_FOUR][7][6], 1.0)
        self.assertEqual(planes[PLANE_SELF_OPEN_FOUR][7][10], 1.0)
        self.assertEqual(planes[PLANE_SELF_OPEN_FOUR].sum(), 2.0)
        self.assertEqual(planes[PLANE_OTHER_OPEN_FOUR].sum(), 0.0)

    def test_the_threat_planes_ignore_the_far_side(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)]))
        planes = encode(position, WHITE, None)
        self.assertEqual(planes[PLANE_OTHER_OPEN_FOUR][7][6], 1.0)
        self.assertEqual(planes[PLANE_OTHER_OPEN_FOUR][7][10], 1.0)
        self.assertEqual(planes[PLANE_SELF_OPEN_FOUR].sum(), 0.0)

    def test_the_mask_marks_the_legal_cells(self) -> None:
        position = Board(board([(7, 7, BLACK)]))
        mask = legal_mask(position, radius=2)
        self.assertEqual(len(mask), ACTION_COUNT)
        self.assertEqual(mask[cell_of(7, 7)], 0.0)
        self.assertEqual(mask[cell_of(7, 8)], 1.0)

    def test_the_symmetries_keep_the_stones_on_the_board(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        planes = encode(position, BLACK, None)
        for index in range(8):
            turned = transform_planes(planes, index)
            stones = sum(1 for plane in turned for row in plane for value in row if value)
            self.assertEqual(stones, 2, index)

    def test_the_threat_planes_follow_the_symmetries(self) -> None:
        position = Board(
            board([(7, 6, BLACK), (7, 7, BLACK), (7, 9, BLACK), (8, 8, WHITE)])
        )
        planes = encode(position, BLACK, (7, 9))
        for index in range(8):
            turned = transform_planes(planes, index)
            moved = permute_board(position, index)
            expected = encode(moved, BLACK, permute_cell((7, 9), index))
            np.testing.assert_allclose(turned, expected, err_msg=str(index))

    def test_the_policy_follows_the_symmetry(self) -> None:
        policy = np.zeros(ACTION_COUNT, dtype=np.float32)
        policy[cell_of(7, 7)] = 1.0
        turned = transform_policy(policy, 1)
        self.assertAlmostEqual(float(turned.sum()), 1.0, places=5)
        self.assertEqual(int(np.argmax(turned)), cell_of(7, 14 - 7))

    def test_the_eight_symmetries_are_different(self) -> None:
        images = {
            tuple(forward(cell) for cell in range(ACTION_COUNT))
            for forward, _ in TRANSFORMS
        }
        self.assertEqual(len(images), 8)

    def test_the_symmetries_form_a_group(self) -> None:
        maps = [forward for forward, _ in TRANSFORMS]
        images = {
            tuple(forward(cell) for cell in range(ACTION_COUNT))
            for forward in maps
        }
        for first in maps:
            for second in maps:
                composed = tuple(first(second(cell)) for cell in range(ACTION_COUNT))
                self.assertIn(composed, images)


class SearchTests(unittest.TestCase):
    def test_the_search_returns_a_legal_move(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=32)
        move = result.best_move()
        self.assertIsNotNone(move)
        row, col = row_col_of(move)
        self.assertEqual(position.grid[row][col], 0)

    def test_the_visits_sum_to_the_simulations(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=40)
        self.assertEqual(sum(result.visits.values()), result.simulations)

    def test_a_five_is_found_without_the_network(self) -> None:
        position = Board(
            board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK), (6, 6, WHITE)])
        )
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=64)
        row, col = row_col_of(result.best_move())
        self.assertIn((row, col), {(7, 6), (7, 11)})

    def test_the_policy_is_a_distribution_over_the_moves(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=32)
        self.assertAlmostEqual(sum(result.policy), 1.0, places=5)

    def test_the_time_budget_stops_the_search(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=10_000_000, time_budget=0.05)
        self.assertLess(result.simulations, 10_000_000)

    def test_the_search_keeps_the_top_candidates_only(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        searcher = MCTS(small_model(), DEVICE, batch_size=8, candidate_width=4)
        result = searcher.search(position, BLACK, simulations=32)
        self.assertLessEqual(len(result.visits), 4)


class ValueTests(unittest.TestCase):
    def test_the_value_changes_its_sign_on_every_level(self) -> None:
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        root = Node(Board(), BLACK)
        middle = Node(Board(), WHITE)
        leaf = Node(Board(), BLACK)
        root.children[0] = middle
        middle.children[0] = leaf

        searcher._backpropagate([root, middle, leaf], 0.7)

        self.assertAlmostEqual(leaf.value_sum, 0.7, places=6)

        self.assertAlmostEqual(middle.value_sum, -0.7, places=6)

        self.assertAlmostEqual(root.value_sum, 0.7, places=6)

    def test_the_parent_reads_the_value_of_the_child_the_other_way_round(self) -> None:
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        node = Node(Board(), BLACK)
        winning = Node(Board(), WHITE)
        losing = Node(Board(), WHITE)


        winning.value_sum = -0.5
        winning.visits = 1
        losing.value_sum = 0.5
        losing.visits = 1
        node.children[0] = winning
        node.children[1] = losing
        winning.prior = 0.5
        losing.prior = 0.5
        self.assertEqual(searcher._select_child(node, set()), 0)

    def test_the_root_value_is_the_value_of_the_root(self) -> None:
        searcher = MCTS(small_model(), DEVICE, batch_size=8)
        root = Node(Board(), BLACK)
        leaf = Node(Board(), WHITE)
        root.children[0] = leaf
        leaf.visits = 4
        searcher._backpropagate([root, leaf], 0.5)
        result = searcher._collect(root)


        self.assertAlmostEqual(result.root_value, -0.5, places=6)


class EngineTests(unittest.TestCase):
    def test_the_engine_takes_an_immediate_win(self) -> None:
        position = board(
            [(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK), (8, 8, WHITE)]
        )
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        move = engine.choose_move(position, [], BLACK, None, {})
        self.assertIn(move, {(7, 6), (7, 11)})

    def test_the_engine_blocks_the_only_five_of_the_opponent(self) -> None:
        position = board(
            [
                (7, 6, BLACK),
                (7, 7, WHITE),
                (7, 8, WHITE),
                (7, 9, WHITE),
                (7, 10, WHITE),
                (8, 8, BLACK),
            ]
        )
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        move = engine.choose_move(position, [], BLACK, None, {})
        self.assertEqual(move, (7, 11))

    def test_the_engine_blocks_one_of_two_fives(self) -> None:
        position = board(
            [
                (7, 6, WHITE),
                (7, 7, WHITE),
                (7, 8, WHITE),
                (7, 9, WHITE),
                (8, 8, BLACK),
            ]
        )
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        move = engine.choose_move(position, [], BLACK, None, {})
        self.assertIn(move, {(7, 5), (7, 10)})

    def test_the_engine_wins_before_it_blocks(self) -> None:
        position = board(
            [
                (7, 7, BLACK),
                (7, 8, BLACK),
                (7, 9, BLACK),
                (7, 10, BLACK),
                (8, 8, WHITE),
                (8, 9, WHITE),
                (8, 10, WHITE),
                (8, 11, WHITE),
            ]
        )
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        move = engine.choose_move(position, [], BLACK, None, {})
        self.assertIn(move, {(7, 6), (7, 11)})

    def test_the_engine_starts_in_the_center(self) -> None:
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        self.assertEqual(engine.choose_move(board([]), [], BLACK, None, {}), (7, 7))

    def test_the_engine_refuses_a_full_board(self) -> None:
        grid = [[BLACK if (row + col) % 2 == 0 else WHITE for col in range(15)] for row in range(15)]
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=8)
        with self.assertRaises(EngineError):
            engine.choose_move(grid, [], BLACK, None, {})

    def test_the_engine_plays_a_legal_move(self) -> None:
        position = board([(7, 7, BLACK), (7, 8, WHITE)])
        engine = AlphaZeroEngine(model_path="models/az/does-not-exist.pt", device="cpu", simulations=16)
        row, col = engine.choose_move(position, [], BLACK, None, {})
        self.assertEqual(position[row][col], 0)

    def test_a_positive_temperature_samples_the_visits(self) -> None:
        position = board([(7, 7, BLACK), (7, 8, WHITE)])
        engine = AlphaZeroEngine(
            model_path="models/az/does-not-exist.pt", device="cpu", simulations=16, temperature=1.0
        )
        for _ in range(5):
            row, col = engine.choose_move(position, [], BLACK, None, {})
            self.assertEqual(position[row][col], 0)


class SelfPlayTests(unittest.TestCase):
    def test_a_game_produces_examples(self) -> None:
        player = SelfPlay(small_model(), DEVICE, simulations=8, batch_size=8)
        game = player.play_game()
        self.assertGreater(len(game.examples), 0)
        for example in game.examples:
            self.assertEqual(len(example.policy), ACTION_COUNT)
            self.assertAlmostEqual(sum(example.policy), 1.0, places=5)
            self.assertIn(example.value, (-1.0, 0.0, 1.0))

    def test_the_examples_of_a_won_game_carry_the_result(self) -> None:
        player = SelfPlay(small_model(), DEVICE, simulations=8, batch_size=8)
        game = player.play_game()
        if game.winner is None:
            self.skipTest("这一局没有分出胜负")
        values = {example.value for example in game.examples}
        self.assertIn(1.0, values)
        self.assertIn(-1.0, values)

    def test_the_augmentation_keeps_the_stones(self) -> None:
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        policy = np.zeros(ACTION_COUNT, dtype=np.float32)
        policy[cell_of(7, 7)] = 1.0
        example = Example(
            planes=encode(position, BLACK, None),
            policy=policy,
            legal=legal_mask(position, 2),
            value=1.0,
        )
        rng = random.Random(3)
        for _ in range(10):
            planes, policy, legal = augment(example, rng)
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
            self.assertAlmostEqual(
                float(legal.sum()), float(example.legal.sum()), places=5
            )

    def test_the_training_lowers_the_loss(self) -> None:
        model = small_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        position = Board(board([(7, 7, BLACK), (7, 8, WHITE)]))
        policy = np.zeros(ACTION_COUNT, dtype=np.float32)
        policy[cell_of(7, 8)] = 1.0
        example = Example(
            planes=encode(position, BLACK, None),
            policy=policy,
            legal=legal_mask(position, 2),
            value=1.0,
        )
        buffer = ReplayBuffer(capacity=10)
        for _ in range(20):
            buffer.add(example)
        losses = train(
            model,
            optimizer,
            buffer,
            DEVICE,
            steps=30,
            batch_size=8,
            augment_data=False,
            rng=random.Random(1),
        )
        self.assertLess(losses[-1], losses[0])


class SettingsTests(unittest.TestCase):
    def test_reads_the_search_parameters(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_ENGINE_TOKEN": "token",
                "AZ_SIMULATIONS": "128",
                "AZ_C_PUCT": "2.0",
                "AZ_MODEL_PATH": "models/az/test.pt",
                "AZ_CANDIDATE_WIDTH": "8",
                "AZ_TIME_BUDGET": "1.5",
            }
        )
        self.assertEqual(settings.simulations, 128)
        self.assertEqual(settings.c_puct, 2.0)
        self.assertEqual(settings.model_path, "models/az/test.pt")
        self.assertEqual(settings.candidate_width, 8)
        self.assertEqual(settings.time_budget, 1.5)
        self.assertEqual(settings.connection.engine_id, "az")


class ThreatTests(unittest.TestCase):
    """威胁平面必须和逐格枚举的结果一致。"""

    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def _windows_through(self, row: int, col: int, dr: int, dc: int, length: int):
        for offset in range(length):
            start_row = row - offset * dr
            start_col = col - offset * dc
            cells = [(start_row + k * dr, start_col + k * dc) for k in range(length)]
            if all(0 <= r < 15 and 0 <= c < 15 for r, c in cells):
                yield cells

    def _reference(self, board: Board, player: int) -> tuple[set, set, set, set]:
        foe = WHITE if player == BLACK else BLACK
        results = []
        for side in (player, foe):
            fives: set = set()
            fours: set = set()
            for row in range(15):
                for col in range(15):
                    if board.grid[row][col] != 0:
                        continue
                    grid = [line[:] for line in board.grid]
                    grid[row][col] = side
                    for dr, dc in self.DIRECTIONS:
                        for cells in self._windows_through(row, col, dr, dc, 5):
                            if all(grid[r][c] == side for r, c in cells):
                                fives.add((row, col))
                                break
                        for cells in self._windows_through(row, col, dr, dc, 6):
                            values = [grid[r][c] for r, c in cells]
                            if values[0] == 0 and values[5] == 0:
                                if all(v == side for v in values[1:5]):
                                    fours.add((row, col))
                                    break
            results.append((fives, fours))
        return results[0][0], results[1][0], results[0][1], results[1][1]

    def test_the_threat_planes_match_a_plain_enumeration(self) -> None:
        rng = random.Random(19)
        for _ in range(6):
            grid = [[0] * 15 for _ in range(15)]
            cells = rng.sample(
                [(r, c) for r in range(15) for c in range(15)], rng.randint(4, 45)
            )
            for row, col in cells:
                grid[row][col] = rng.choice([BLACK, WHITE])
            board = Board(grid)
            for player in (BLACK, WHITE):
                planes = encode(board, player, None)
                self_five, other_five, self_four, other_four = self._reference(board, player)
                self.assertEqual(
                    {tuple(cell) for cell in np.argwhere(planes[PLANE_SELF_FIVE] > 0)},
                    self_five,
                )
                self.assertEqual(
                    {tuple(cell) for cell in np.argwhere(planes[PLANE_OTHER_FIVE] > 0)},
                    other_five,
                )
                self.assertEqual(
                    {tuple(cell) for cell in np.argwhere(planes[PLANE_SELF_OPEN_FOUR] > 0)},
                    self_four,
                )
                self.assertEqual(
                    {tuple(cell) for cell in np.argwhere(planes[PLANE_OTHER_OPEN_FOUR] > 0)},
                    other_four,
                )


class CheckpointTests(unittest.TestCase):
    def test_an_old_checkpoint_still_loads(self) -> None:
        model = small_model()
        state = dict(model.state_dict())
        state["stem.weight"] = state["stem.weight"][:, :3].contiguous()
        path = Path("models/az/test-old-planes.pt")
        try:
            torch.save(
                {"state": state, "channels": 8, "blocks": 1, "value_hidden": 8}, path
            )
            loaded = load_model(str(path), DEVICE)
            self.assertEqual(loaded.stem.weight.shape[1], PLANE_COUNT)
            x = torch.zeros(1, PLANE_COUNT, 15, 15)
            x[:, 0, 7, 7] = 1.0
            model.eval()
            loaded.eval()
            self.assertTrue(
                torch.allclose(model(x)[0], loaded(x)[0], atol=1e-5),
            )
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
