from __future__ import annotations

import math
import random
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from az_worker.board import BLACK, BOARD_SIZE, WHITE, Board, other
from az_worker.config import Settings
from az_worker.dataset import TensorDataset
from az_worker.engine import AlphaZeroEngine, EngineError
from az_worker.features import (
    CELL_PERMUTATIONS,
    PLANE_COUNT,
    PLANE_LAST,
    PLANE_OTHER,
    PLANE_OTHER_FIVE,
    PLANE_OTHER_OPEN_FOUR,
    PLANE_OTHER_OPEN_THREE,
    PLANE_SELF,
    PLANE_SELF_FIVE,
    PLANE_SELF_OPEN_FOUR,
    PLANE_SELF_OPEN_THREE,
    TRANSFORMS,
    encode,
    legal_mask,
)
from az_worker.mcts import C_BASE, C_INIT, MCTS, Node, cell_of, row_col_of
from az_worker.network import (
    ACTION_COUNT,
    GomokuNet,
    load_model,
    read_extra,
    read_meta,
    save,
    train_batch,
)
from az_worker.selfplay import (
    Example,
    SelfPlay,
    checkpoint_stamp,
    worker_init,
    worker_play,
)
from az_worker.threats import board_masks, five_mask, open_three_mask, threat_masks
from az_worker.train import (
    append_iteration_log,
    learning_rate,
    plan_games,
    plan_workers,
    simulations_for,
)
from az_worker.vcf import VCF, VCT, Solver, winning_move


DEVICE = torch.device("cpu")


def board_of(stones: list[tuple[int, int, int]]) -> list[list[int]]:
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for row, col, player in stones:
        grid[row][col] = player
    return grid


def small_model() -> GomokuNet:
    return GomokuNet(channels=8, blocks=1, value_hidden=8)


class StubNet(torch.nn.Module):
    def __init__(self, policy: np.ndarray) -> None:
        super().__init__()
        self.policy = np.log(policy + 1e-9).astype(np.float32)

    def forward(self, x: torch.Tensor):
        logits = torch.from_numpy(self.policy).expand(x.shape[0], -1)
        return logits, torch.zeros(x.shape[0])


def permute_planes(planes: np.ndarray, index: int) -> np.ndarray:
    flat = planes.reshape(planes.shape[0], BOARD_SIZE * BOARD_SIZE)
    return flat[:, CELL_PERMUTATIONS[index]].reshape(planes.shape)


def permute_board(position: Board, index: int) -> Board:
    forward, _ = TRANSFORMS[index]
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for row, col, stone in position.stones:
        new_row, new_col = divmod(forward(row * BOARD_SIZE + col), BOARD_SIZE)
        grid[new_row][new_col] = stone
    return Board(grid)


def plane_cells(planes: np.ndarray, plane: int) -> set[tuple[int, int]]:
    cells = np.argwhere(planes[plane] > 0.5)
    return {(int(row), int(col)) for row, col in cells}


class BoardTests(unittest.TestCase):
    def test_recognises_a_five(self) -> None:
        position = Board(board_of([(7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (7, 8, BLACK)]))
        self.assertTrue(position.wins_with(7, 9, BLACK))
        self.assertTrue(position.play(7, 9, BLACK))

    def test_empty_cells_exclude_stones(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (6, 6, WHITE)]))
        cells = position.empty_cells()
        self.assertEqual(len(cells), BOARD_SIZE * BOARD_SIZE - 2)
        self.assertNotIn((7, 7), cells)
        self.assertIn((7, 8), cells)


class FeatureTests(unittest.TestCase):
    def test_nine_planes(self) -> None:
        self.assertEqual(PLANE_COUNT, 9)
        planes = encode(Board(), BLACK, None)
        self.assertEqual(planes.shape, (9, BOARD_SIZE, BOARD_SIZE))

    def test_the_planes_follow_the_side_to_move(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (6, 6, WHITE)]))
        black = encode(position, BLACK, (6, 6))
        white = encode(position, WHITE, (6, 6))
        self.assertEqual(plane_cells(black, PLANE_SELF), {(7, 7)})
        self.assertEqual(plane_cells(black, PLANE_OTHER), {(6, 6)})
        self.assertEqual(plane_cells(white, PLANE_SELF), {(6, 6)})
        self.assertEqual(plane_cells(white, PLANE_OTHER), {(7, 7)})
        self.assertEqual(plane_cells(black, PLANE_LAST), {(6, 6)})

    def test_the_five_plane_marks_the_winning_cells(self) -> None:
        position = Board(board_of([(7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (7, 8, BLACK)]))
        planes = encode(position, BLACK, None)
        self.assertEqual(plane_cells(planes, PLANE_SELF_FIVE), {(7, 4), (7, 9)})

    def test_the_open_four_plane_marks_the_live_four_points(self) -> None:
        position = Board(board_of([(7, 6, BLACK), (7, 7, BLACK), (7, 8, BLACK)]))
        planes = encode(position, BLACK, None)
        self.assertEqual(plane_cells(planes, PLANE_SELF_OPEN_FOUR), {(7, 5), (7, 9)})

    def test_the_open_three_plane_marks_the_live_three_points(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (7, 8, BLACK)]))
        planes = encode(position, BLACK, None)
        self.assertEqual(
            plane_cells(planes, PLANE_SELF_OPEN_THREE),
            {(7, 6), (7, 9)},
        )

    def test_the_open_three_plane_is_zero_when_blocked(self) -> None:
        position = Board(
            board_of([(7, 6, WHITE), (7, 7, BLACK), (7, 8, BLACK), (7, 10, WHITE)])
        )
        planes = encode(position, BLACK, None)
        self.assertEqual(plane_cells(planes, PLANE_SELF_OPEN_THREE), set())

    def test_the_threat_planes_match_a_plain_enumeration(self) -> None:
        rng = random.Random(11)
        for _ in range(25):
            stones: list[tuple[int, int, int]] = []
            cells = rng.sample(range(BOARD_SIZE * BOARD_SIZE), rng.randrange(4, 18))
            for index in cells:
                row, col = divmod(index, BOARD_SIZE)
                stones.append((row, col, rng.choice((BLACK, WHITE))))
            position = Board(board_of(stones))
            planes = encode(position, BLACK, None)
            self.assertEqual(
                plane_cells(planes, PLANE_SELF_FIVE), reference_five_points(position, BLACK)
            )
            self.assertEqual(
                plane_cells(planes, PLANE_OTHER_FIVE), reference_five_points(position, WHITE)
            )
            self.assertEqual(
                plane_cells(planes, PLANE_SELF_OPEN_FOUR),
                reference_open_four_points(position, BLACK),
            )
            self.assertEqual(
                plane_cells(planes, PLANE_SELF_OPEN_THREE),
                reference_open_three_points(position, BLACK),
            )

    def test_the_planes_follow_the_symmetries(self) -> None:
        position = Board(
            board_of([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 7, WHITE), (5, 9, BLACK)])
        )
        planes = encode(position, BLACK, (5, 9))
        for index in range(len(TRANSFORMS)):
            self.assertTrue(
                np.array_equal(permute_planes(planes, index), encode(permute_board(position, index), BLACK, permute_cell((5, 9), index))),
                f"transform {index}",
            )

    def test_the_mask_marks_every_empty_cell(self) -> None:
        position = Board(board_of([(7, 7, BLACK)]))
        mask = legal_mask(position)
        self.assertEqual(int(mask.sum()), BOARD_SIZE * BOARD_SIZE - 1)
        self.assertEqual(float(mask[cell_of(7, 7)]), 0.0)

    def test_the_plane_count_is_enforced(self) -> None:
        from az_worker.features import as_planes

        with self.assertRaises(ValueError):
            as_planes(np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32))


def permute_cell(cell: tuple[int, int], index: int) -> tuple[int, int]:
    forward, _ = TRANSFORMS[index]
    return divmod(forward(cell[0] * BOARD_SIZE + cell[1]), BOARD_SIZE)


def reference_five_points(position: Board, player: int) -> set[tuple[int, int]]:
    return {
        (row, col)
        for row, col in position.empty_cells()
        if position.wins_with(row, col, player)
    }


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _run(
    position: Board, row: int, col: int, player: int, dr: int, dc: int
) -> tuple[int, list[tuple[int, int]]]:
    length = 1
    ends = []
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and position.grid[r][c] == player:
            length += 1
            r += sign * dr
            c += sign * dc
        ends.append((r, c))
    return length, ends


def _open_run(position: Board, row: int, col: int, player: int, size: int) -> bool:
    for dr, dc in DIRECTIONS:
        length, ends = _run(position, row, col, player, dr, dc)
        if length != size:
            continue
        if all(
            0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and position.grid[r][c] == 0
            for r, c in ends
        ):
            return True
    return False


def reference_open_four_points(position: Board, player: int) -> set[tuple[int, int]]:
    found = set()
    for row, col in position.empty_cells():
        position.grid[row][col] = player
        try:
            if _open_run(position, row, col, player, 4):
                found.add((row, col))
        finally:
            position.grid[row][col] = 0
    return found


def reference_open_three_points(position: Board, player: int) -> set[tuple[int, int]]:
    found = set()
    for row, col in position.empty_cells():
        position.grid[row][col] = player
        try:
            if _open_run(position, row, col, player, 3):
                found.add((row, col))
        finally:
            position.grid[row][col] = 0
    return found


class ThreatBitboardTests(unittest.TestCase):
    def test_the_masks_agree_with_the_planes(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE)]))
        own, other, empty = board_masks(position, BLACK)
        self.assertEqual(five_mask(own, empty) & ~empty, 0)
        masks = threat_masks(own, other, empty)
        self.assertEqual(len(masks), 6)
        self.assertTrue(open_three_mask(own, empty))


class SelfPlaySolverTests(unittest.TestCase):
    def test_the_solver_seats_a_forced_win(self) -> None:
        from az_worker.vcf import PlaySolver

        solver = PlaySolver(vcf_budget=0.1, vct_budget=0.0)
        position = Board(
            board_of(
                [
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (7, 7, BLACK),
                    (6, 6, WHITE),
                    (8, 8, WHITE),
                ]
            )
        )
        self.assertIn(solver.move(position, BLACK), {(7, 4), (7, 8)})

    def test_selfplay_uses_one_hot_targets_on_solver_plies(self) -> None:
        class FirstEmpty:
            def move(self, board, player):
                return board.empty_cells()[0]

        player = SelfPlay(
            small_model(), DEVICE, simulations=2, batch_size=2, seed=1, solver=FirstEmpty()
        )
        game = player.play_game()
        self.assertGreater(game.moves, 0)
        for example in game.examples:
            self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=6)
            self.assertEqual(int((example.policy > 0).sum()), 1)

    def test_selfplay_without_a_solver_keeps_the_visit_target(self) -> None:
        player = SelfPlay(small_model(), DEVICE, simulations=4, batch_size=2, seed=1)
        game = player.play_game()
        visited = [int((example.policy > 0).sum()) for example in game.examples]
        self.assertGreater(max(visited), 1)


class PuctTests(unittest.TestCase):
    def test_the_exploration_constant_grows_with_visits(self) -> None:
        low = math.log((1 + 0 + C_BASE) / C_BASE) + C_INIT
        high = math.log((1 + 1_000_000 + C_BASE) / C_BASE) + C_INIT
        self.assertAlmostEqual(low, C_INIT, places=3)
        self.assertGreater(high, low)
        self.assertAlmostEqual(C_BASE, 19652.0)
        self.assertAlmostEqual(C_INIT, 1.25)

    def test_the_selection_matches_the_puct_formula(self) -> None:
        model = small_model()
        searcher = MCTS(model, DEVICE)
        root = Node(Board(), BLACK)
        root.visits = 40
        moves = [(7, 7), (7, 8)]
        priors = [0.7, 0.3]
        visits = [12, 28]
        values = [-0.4, 0.6]
        for (row, col), prior, count, value in zip(moves, priors, visits, values):
            child = Node(None, WHITE, parent=root, move=(row, col))
            child.prior = prior
            child.visits = count
            child.value_sum = value * count
            root.children[cell_of(row, col)] = child

        constant = math.log((1 + root.visits + C_BASE) / C_BASE) + C_INIT
        sqrt_total = math.sqrt(root.visits)
        expected = []
        for (row, col), prior, count, value in zip(moves, priors, visits, values):
            quality = -value
            expected.append(quality + constant * prior * sqrt_total / (1 + count))
        chosen = searcher._select_child(root, set())
        self.assertEqual(chosen, cell_of(*moves[int(np.argmax(expected))]))

    def test_virtual_loss_moves_the_search_elsewhere(self) -> None:
        model = small_model()
        searcher = MCTS(model, DEVICE)
        root = Node(Board(), BLACK)
        root.visits = 10
        for row, col in ((7, 7), (7, 8)):
            child = Node(None, WHITE, parent=root, move=(row, col))
            child.prior = 0.5
            root.children[cell_of(row, col)] = child
        self.assertEqual(searcher._select_child(root, set()), cell_of(7, 7))
        root.children[cell_of(7, 7)].virtual_loss = 3
        self.assertEqual(searcher._select_child(root, set()), cell_of(7, 8))

    def test_the_value_changes_its_sign_on_every_level(self) -> None:
        model = small_model()
        searcher = MCTS(model, DEVICE)
        root = Node(Board(), BLACK)
        child = Node(None, WHITE, parent=root, move=(7, 7))
        root.children[cell_of(7, 7)] = child
        root.expanded = True
        child.expanded = True
        searcher._backpropagate([root, child], 1.0)
        self.assertEqual(child.value_sum, 1.0)
        self.assertEqual(root.value_sum, -1.0)


class SearchTests(unittest.TestCase):
    def test_the_visits_sum_to_the_simulations(self) -> None:
        model = small_model()
        position = Board(board_of([(7, 7, BLACK), (6, 8, WHITE)]))
        result = MCTS(model, DEVICE, batch_size=8).search(
            position, BLACK, simulations=48, last_move=(6, 8)
        )
        self.assertEqual(sum(result.visits.values()), 48)

    def test_every_legal_move_is_a_child(self) -> None:
        model = small_model()
        position = Board(board_of([(7, 7, BLACK)]))
        result = MCTS(model, DEVICE, batch_size=8).search(position, WHITE, simulations=16)
        self.assertEqual(len(result.visits), BOARD_SIZE * BOARD_SIZE - 1)

    def test_the_policy_is_a_distribution_over_the_moves(self) -> None:
        model = small_model()
        position = Board(board_of([(7, 7, BLACK)]))
        result = MCTS(model, DEVICE, batch_size=8).search(position, WHITE, simulations=16)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=5)
        visited = sum(1 for count in result.visits.values() if count > 0)
        self.assertEqual(int((result.policy > 0).sum()), visited)
        self.assertLess(visited, len(result.visits))

    def test_the_search_converts_a_prior_into_a_five(self) -> None:
        position = Board(
            board_of([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])
        )
        policy = np.full(ACTION_COUNT, 1e-4, dtype=np.float32)
        policy[cell_of(7, 3)] = 0.4999
        policy[cell_of(7, 8)] = 0.4999
        searcher = MCTS(StubNet(policy), DEVICE, batch_size=8)
        result = searcher.search(position, BLACK, simulations=64)
        self.assertIn(result.best_move(), {cell_of(7, 3), cell_of(7, 8)})

    def test_the_time_budget_stops_the_search(self) -> None:
        model = small_model()
        position = Board(board_of([(7, 7, BLACK)]))
        started = time.monotonic()
        result = MCTS(model, DEVICE, batch_size=8).search(
            position, WHITE, simulations=1_000_000, time_budget=0.3
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertLess(sum(result.visits.values()), 1_000_000)

    def test_dirichlet_noise_shifts_the_root_priors(self) -> None:
        model = small_model()
        searcher = MCTS(model, DEVICE, batch_size=8, seed=3)
        root = Node(Board(), BLACK)
        searcher._expand(root, np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float32))
        before = np.array([child.prior for child in root.children.values()])
        searcher._add_root_noise(root)
        after = np.array([child.prior for child in root.children.values()])
        self.assertAlmostEqual(float(after.sum()), 1.0, places=6)
        self.assertGreater(float(np.abs(after - before).max()), 1e-3)

    def test_a_root_child_that_wins_is_terminal(self) -> None:
        model = small_model()
        position = Board(
            board_of([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])
        )
        root = Node(position, BLACK)
        searcher = MCTS(model, DEVICE)
        searcher._expand(root, np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float32))
        self.assertEqual(root.children[cell_of(7, 8)].terminal_value, -1.0)
        self.assertIsNone(root.children[cell_of(0, 0)].terminal_value)

    def test_the_last_empty_cell_is_a_draw(self) -> None:
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        empties = 0
        last = (0, 0)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (row + 2 * col) % 4 in (0, 1):
                    grid[row][col] = BLACK
                else:
                    grid[row][col] = WHITE
                empties += 1
        grid[0][0] = 0
        position = Board(grid)
        root = Node(position, BLACK)
        searcher = MCTS(small_model(), DEVICE)
        searcher._expand(root, np.zeros(ACTION_COUNT, dtype=np.float32))
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[cell_of(0, 0)].terminal_value, 0.0)

    def test_a_positive_temperature_samples_the_visits(self) -> None:
        from az_worker.mcts import SearchResult

        result = SearchResult({0: 1, 1: 1, 2: 100}, np.zeros(ACTION_COUNT), 0.0, 102)
        rng = random.Random(2)
        picks = [result.sampled_move(1.0, rng) for _ in range(200)]
        self.assertEqual(set(picks), {0, 1, 2})
        self.assertEqual(result.sampled_move(0.0, rng), 2)


class SelfPlayTests(unittest.TestCase):
    def test_a_game_produces_examples(self) -> None:
        player = SelfPlay(small_model(), DEVICE, simulations=6, batch_size=4, seed=1)
        game = player.play_game()
        self.assertGreater(game.moves, 0)
        self.assertEqual(len(game.examples), game.moves)
        for example in game.examples:
            self.assertEqual(example.planes.shape, (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))
            self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=5)

    def test_the_examples_of_a_won_game_carry_the_result(self) -> None:
        player = SelfPlay(small_model(), DEVICE, simulations=8, batch_size=4, seed=2)
        game = player.play_game()
        if game.winner is None:
            for example in game.examples:
                self.assertEqual(example.value, 0.0)
            return
        values = {example.value for example in game.examples}
        self.assertTrue(values <= {1.0, -1.0})
        self.assertIn(1.0, values)
        self.assertIn(-1.0, values)


class ScheduleTests(unittest.TestCase):
    def test_the_learning_rate_decays_three_times(self) -> None:
        total = 25_000
        self.assertAlmostEqual(learning_rate(0, total), 0.0125)
        self.assertAlmostEqual(learning_rate(3_570, total), 0.0125)
        self.assertAlmostEqual(learning_rate(3_571, total), 0.00125)
        self.assertAlmostEqual(learning_rate(10_713, total), 0.00125)
        self.assertAlmostEqual(learning_rate(10_714, total), 0.000125)
        self.assertAlmostEqual(learning_rate(17_856, total), 0.000125)
        self.assertAlmostEqual(learning_rate(17_857, total), 0.0000125)
        self.assertAlmostEqual(learning_rate(24_999, total), 0.0000125)

    def test_the_simulations_ramp_in_thirds(self) -> None:
        total = 25_000
        self.assertEqual(simulations_for(0, total), 128)
        self.assertEqual(simulations_for(8_332, total), 128)
        self.assertEqual(simulations_for(8_334, total), 256)
        self.assertEqual(simulations_for(16_666, total), 256)
        self.assertEqual(simulations_for(16_667, total), 400)
        self.assertEqual(simulations_for(24_999, total), 400)


class ParallelSelfPlayTests(unittest.TestCase):
    def test_a_worker_plays_from_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save(small_model(), str(path))
            worker_init("cpu", 12)
            game = worker_play((str(path), checkpoint_stamp(str(path)), 6, 4, 1))
        self.assertGreater(game.moves, 0)
        for example in game.examples:
            self.assertEqual(
                example.planes.shape, (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
            )
            self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=5)

    def test_two_processes_return_games(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save(small_model(), str(path))
            stamp = checkpoint_stamp(str(path))
            with ProcessPoolExecutor(
                max_workers=2, initializer=worker_init, initargs=("cpu", 12)
            ) as pool:
                futures = [
                    pool.submit(worker_play, (str(path), stamp, 6, 4, seed))
                    for seed in (1, 2)
                ]
                games = [future.result() for future in futures]
        self.assertEqual(len(games), 2)
        for game in games:
            self.assertGreater(game.moves, 0)
            for example in game.examples:
                self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=5)


class WorkerPlanTests(unittest.TestCase):
    def test_a_fixed_count_is_kept(self) -> None:
        self.assertEqual(plan_workers("4", 8 * 1024**3, 32), 4)
        self.assertEqual(plan_workers(1, None, 32), 1)

    def test_auto_fits_the_free_vram(self) -> None:
        self.assertEqual(plan_workers("auto", 4 * 1024**3, 32), 13)

    def test_auto_never_exceeds_half_the_cores(self) -> None:
        self.assertEqual(plan_workers("auto", 64 * 1024**3, 8), 4)

    def test_auto_without_a_gpu_uses_eight_cpu_workers(self) -> None:
        self.assertEqual(plan_workers("auto", None, 32), 8)
        self.assertEqual(plan_workers("auto", None, 4), 2)

    def test_auto_games_keep_the_pool_busy(self) -> None:
        self.assertEqual(plan_games("auto", 13), 13)
        self.assertEqual(plan_games("auto", 4), 10)
        self.assertEqual(plan_games("7", 13), 7)


class SharedInferenceTests(unittest.TestCase):
    def test_a_worker_plays_through_the_inference_server(self) -> None:
        import multiprocessing as mp
        import threading

        from az_worker.inference import RemoteEvaluator, serve

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "model.pt")
            save(small_model(), path)
            requests: mp.Queue = mp.Queue()
            stop = threading.Event()
            thread = threading.Thread(
                target=serve,
                args=(requests, "cpu", path, stop),
                kwargs={"max_batch": 8, "window": 0.001},
                daemon=True,
            )
            thread.start()
            parent, child = mp.Pipe()
            requests.put(parent)
            evaluator = RemoteEvaluator(child, timeout=30.0)
            player = SelfPlay(
                None,
                torch.device("cpu"),
                simulations=4,
                batch_size=4,
                seed=1,
                evaluator=evaluator,
            )
            game = player.play_game()
            stop.set()
            thread.join(5.0)
        self.assertGreater(game.moves, 0)
        for example in game.examples:
            self.assertAlmostEqual(float(example.policy.sum()), 1.0, places=5)
        self.assertTrue(stop.is_set())


class CheckpointExtraTests(unittest.TestCase):
    def test_the_optimizer_state_round_trips(self) -> None:
        model = small_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        planes = torch.zeros(2, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        policy = torch.full((2, ACTION_COUNT), 1.0 / ACTION_COUNT)
        mask = torch.ones((2, ACTION_COUNT))
        value = torch.ones(2)
        train_batch(model, optimizer, planes, policy, value, mask)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "model.pt")
            save(model, path, {"step": 5, "games": 7}, {"optimizer": optimizer.state_dict()})
            extra = read_extra(path)
            meta = read_meta(path)
            restored_model = load_model(path, DEVICE)
        self.assertEqual(int(meta["step"]), 5)
        self.assertEqual(int(meta["games"]), 7)
        self.assertIn("optimizer", extra)
        restored = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
        restored.load_state_dict(extra["optimizer"])
        self.assertEqual(len(restored.state), len(optimizer.state))


class TrainingLogTests(unittest.TestCase):
    def test_the_log_writes_one_header_then_a_row_per_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "train.log"
            row = {
                "time": "2026-11-17 18:00:00",
                "iteration": 1,
                "step": 250,
                "learning_rate": 0.0125,
                "simulations": 128,
                "games": 13,
                "total_games": 13,
                "wins": 7,
                "losses": 6,
                "draws": 0,
                "plies": 780,
                "buffer": 780,
                "policy_loss": 4.1234,
                "value_loss": 0.5,
                "steps_per_second": 9.2,
                "seconds": 27.3,
                "eval_winrate": "",
            }
            append_iteration_log(path, row)
            append_iteration_log(path, {**row, "iteration": 2, "step": 500})
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("time,iteration,step,learning_rate"))
        self.assertIn(",250,", lines[1])
        self.assertEqual(len(lines[0].split(",")), len(lines[1].split(",")))


class DatasetTests(unittest.TestCase):
    def test_the_buffer_is_first_in_first_out(self) -> None:
        dataset = TensorDataset(DEVICE, capacity=4)
        examples = []
        for value in range(6):
            planes = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            planes[0, 0, 0] = value
            examples.append(
                Example(
                    planes=planes,
                    policy=np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float32),
                    legal=np.ones(ACTION_COUNT, dtype=np.float32),
                    value=float(value),
                )
            )
        dataset.extend(examples)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.value[:4].tolist(), [2.0, 3.0, 4.0, 5.0])

    def test_sampling_returns_network_ready_tensors(self) -> None:
        dataset = TensorDataset(DEVICE, capacity=8)
        dataset.extend(
            [
                Example(
                    planes=np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                    policy=np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float32),
                    legal=np.ones(ACTION_COUNT, dtype=np.float32),
                    value=1.0,
                )
            ]
        )
        planes, policy, mask, value = dataset.sample(4, random.Random(0))
        self.assertEqual(tuple(planes.shape), (4, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(tuple(policy.shape), (4, ACTION_COUNT))
        self.assertEqual(tuple(mask.shape), (4, ACTION_COUNT))
        self.assertEqual(tuple(value.shape), (4,))

    def test_augmented_sampling_permutes_the_targets(self) -> None:
        dataset = TensorDataset(DEVICE, capacity=4)
        planes = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0, 7, 7] = 1.0
        policy = np.zeros(ACTION_COUNT, dtype=np.float32)
        policy[cell_of(7, 7)] = 1.0
        dataset.extend(
            [
                Example(
                    planes=planes,
                    policy=policy,
                    legal=np.ones(ACTION_COUNT, dtype=np.float32),
                    value=1.0,
                )
            ]
        )
        batch_planes, batch_policy, _, _ = dataset.sample(1, random.Random(1), augment=True)
        self.assertEqual(tuple(batch_planes.shape), (1, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE))
        self.assertAlmostEqual(float(batch_policy.sum()), 1.0, places=6)
        stones = batch_planes[0, 0].argmax().item()
        chosen = int(batch_policy[0].argmax())
        self.assertEqual(stones, chosen)

    def test_a_saved_buffer_round_trips(self) -> None:
        dataset = TensorDataset(DEVICE, capacity=4)
        dataset.extend(
            [
                Example(
                    planes=np.ones((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                    policy=np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float32),
                    legal=np.ones(ACTION_COUNT, dtype=np.float32),
                    value=-1.0,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "replay.pt"
            dataset.save(path)
            loaded = TensorDataset.load(path, DEVICE)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.value[0].item(), -1.0)


class NetworkTests(unittest.TestCase):
    def test_the_policy_and_value_shapes(self) -> None:
        model = small_model()
        planes = torch.zeros(3, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        logits, values = model(planes)
        self.assertEqual(tuple(logits.shape), (3, ACTION_COUNT))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(bool((values.abs() <= 1.0).all()))

    def test_checkpoints_round_trip(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save(model, str(path), {"step": 7})
            loaded = load_model(str(path), DEVICE)
            self.assertEqual(loaded.channels, model.channels)
            self.assertEqual(loaded.value_hidden, model.value_hidden)

    def test_a_seven_plane_checkpoint_is_rejected(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(
                {
                    "state": model.state_dict(),
                    "planes": 7,
                },
                path,
            )
            with self.assertRaises(ValueError):
                load_model(str(path), DEVICE)

    def test_one_training_step_moves_the_loss(self) -> None:
        model = small_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        planes = torch.zeros(4, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE)
        target_policy = torch.full((4, ACTION_COUNT), 1.0 / ACTION_COUNT)
        mask = torch.ones((4, ACTION_COUNT))
        target_value = torch.ones(4)
        first, _, _ = train_batch(model, optimizer, planes, target_policy, target_value, mask)
        for _ in range(20):
            last, policy_loss, value_loss = train_batch(
                model, optimizer, planes, target_policy, target_value, mask
            )
        self.assertLess(last, first)


class SolverTests(unittest.TestCase):
    def test_vcf_finds_a_four_chain(self) -> None:
        position = Board(
            board_of(
                [
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (7, 7, BLACK),
                    (8, 8, WHITE),
                    (6, 9, WHITE),
                ]
            )
        )
        move = winning_move(position, BLACK, 1.0, VCF)
        self.assertIsNotNone(move)
        self.assertIn(move, {(7, 4), (7, 8)})

    def test_vcf_returns_nothing_on_a_quiet_position(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (8, 8, WHITE), (6, 6, BLACK)]))
        self.assertIsNone(winning_move(position, BLACK, 0.5, VCF))

    def test_vcf_ignores_a_position_without_stones(self) -> None:
        self.assertIsNone(winning_move(Board(), BLACK, 0.2, VCF))

    def test_vct_finds_a_double_three(self) -> None:
        position = Board(
            board_of(
                [
                    (7, 8, BLACK),
                    (7, 9, BLACK),
                    (8, 7, BLACK),
                    (9, 7, BLACK),
                    (5, 5, WHITE),
                    (10, 10, WHITE),
                ]
            )
        )
        move = winning_move(position, BLACK, 2.0, VCT)
        self.assertEqual(move, (7, 7))

    def test_the_solver_does_not_mutate_the_board(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (8, 7, BLACK)]))
        before = [row[:] for row in position.grid]
        solver = Solver(max_depth=4, max_nodes=100)
        solver.winning_move(position, BLACK, 0.2, VCF)
        self.assertEqual(before, position.grid)

    def test_the_deadline_aborts_the_search(self) -> None:
        position = Board(
            board_of(
                [
                    (7, 7, BLACK),
                    (7, 8, BLACK),
                    (7, 9, BLACK),
                    (8, 8, WHITE),
                    (6, 6, WHITE),
                    (9, 9, WHITE),
                ]
            )
        )
        move = winning_move(position, BLACK, 0.001, VCF, max_depth=10, max_nodes=10_000)
        self.assertTrue(move is None or isinstance(move, tuple))


class EngineTests(unittest.TestCase):
    def engine(self, **kwargs) -> AlphaZeroEngine:
        return AlphaZeroEngine(
            model_path="models/az/missing_test_checkpoint.pt", device="cpu", **kwargs
        )

    def test_the_engine_takes_an_immediate_win(self) -> None:
        position = Board(
            board_of([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])
        )
        move = self.engine(simulations=4, vcf_budget=0.0, vct_budget=0.0).choose_move(
            [row[:] for row in position.grid], [], BLACK, None, {}
        )
        self.assertIn(move, {(7, 3), (7, 8)})

    def test_the_engine_blocks_the_only_five_of_the_opponent(self) -> None:
        position = Board(
            board_of([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (6, 6, WHITE)])
        )
        move = self.engine(simulations=4, vcf_budget=0.0, vct_budget=0.0).choose_move(
            [row[:] for row in position.grid], [], WHITE, None, {}
        )
        self.assertIn(move, {(7, 3), (7, 8)})

    def test_the_engine_starts_in_the_centre(self) -> None:
        self.assertEqual(self.engine().choose_move(board_of([]), [], BLACK, None, {}), (7, 7))

    def test_the_engine_refuses_a_full_board(self) -> None:
        grid = [[BLACK] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        with self.assertRaises(EngineError):
            self.engine().choose_move(grid, [], BLACK, None, {})

    def test_the_engine_plays_a_legal_move(self) -> None:
        position = Board(board_of([(7, 7, BLACK), (6, 8, WHITE)]))
        move = self.engine(simulations=6, vcf_budget=0.0, vct_budget=0.0).choose_move(
            [row[:] for row in position.grid], [], BLACK, {"row": 6, "col": 8}, {}
        )
        self.assertEqual(position.grid[move[0]][move[1]], 0)

    def test_the_engine_respects_the_total_time_budget(self) -> None:
        position = Board(
            board_of([(7, 7, BLACK), (6, 8, WHITE), (5, 5, BLACK), (9, 9, WHITE)])
        )
        engine = self.engine(
            simulations=1_000_000, time_budget=0.4, vcf_budget=0.1, vct_budget=0.1
        )
        started = time.monotonic()
        move = engine.choose_move([row[:] for row in position.grid], [], BLACK, None, {})
        elapsed = time.monotonic() - started
        self.assertEqual(position.grid[move[0]][move[1]], 0)
        self.assertLess(elapsed, 1.2)

    def test_the_engine_finds_a_forced_win_through_vcf(self) -> None:
        position = Board(
            board_of(
                [
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (7, 7, BLACK),
                    (6, 6, WHITE),
                    (8, 8, WHITE),
                ]
            )
        )
        move = self.engine(simulations=4, vcf_budget=0.5, vct_budget=0.0).choose_move(
            [row[:] for row in position.grid], [], BLACK, None, {}
        )
        self.assertIn(move, {(7, 4), (7, 8)})


class SettingsTests(unittest.TestCase):
    def test_reads_the_solver_parameters(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_URL": "http://127.0.0.1:8080/api/v1",
                "MYGO_ENGINE_TOKEN": "token",
                "MYGO_ENGINE_ID": "az",
                "AZ_VCF_BUDGET": "0.3",
                "AZ_VCT_BUDGET": "0.4",
                "AZ_SOLVER_DEPTH": "8",
                "AZ_SOLVER_NODES": "500",
                "AZ_SIMULATIONS": "100000",
                "AZ_TIME_BUDGET": "2.0",
            }
        )
        self.assertAlmostEqual(settings.vcf_budget, 0.3)
        self.assertAlmostEqual(settings.vct_budget, 0.4)
        self.assertEqual(settings.solver_depth, 8)
        self.assertEqual(settings.solver_nodes, 500)
        self.assertEqual(settings.simulations, 100000)
        self.assertAlmostEqual(settings.time_budget, 2.0)

    def test_the_defaults_keep_vcf_and_vct_on(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_URL": "http://127.0.0.1:8080/api/v1",
                "MYGO_ENGINE_TOKEN": "token",
                "MYGO_ENGINE_ID": "az",
            }
        )
        self.assertGreater(settings.vcf_budget, 0.0)
        self.assertGreater(settings.vct_budget, 0.0)


if __name__ == "__main__":
    unittest.main()
