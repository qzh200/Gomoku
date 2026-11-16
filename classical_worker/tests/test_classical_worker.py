"""Tests of the classical engine.

The tests run with the standard library only:
    python -m unittest discover -s classical_worker/tests -t .
"""

from __future__ import annotations

import unittest

from classical_worker.config import ConfigError, Settings
from classical_worker.engine import ClassicalEngine, EngineError
from classical_worker.position import BLACK, WHITE, WIN_SCORE, Position, coordinate_of, other
from classical_worker.search import INFINITY, Searcher


def board(stones: list[tuple[int, int, int]]) -> list[list[int]]:
    """Return a 15x15 board with the given stones."""
    grid = [[0] * 15 for _ in range(15)]
    for row, col, player in stones:
        grid[row][col] = player
    return grid


def empty() -> list[list[int]]:
    """Return an empty board."""
    return [[0] * 15 for _ in range(15)]


class PositionTests(unittest.TestCase):
    def test_counts_an_open_run_above_a_closed_one(self) -> None:
        # H8 I8 J8 with both ends free, against the same run pinned to the edge.
        open_run = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)]))
        closed_run = Position(board([(0, 0, BLACK), (0, 1, BLACK), (0, 2, BLACK)]))
        self.assertGreater(open_run.evaluate(BLACK), closed_run.evaluate(BLACK))

    def test_scores_the_players_from_their_own_side(self) -> None:
        position = Position(board([(7, 7, BLACK)]))
        self.assertEqual(position.evaluate(BLACK), -position.evaluate(WHITE))

    def test_play_and_undo_restore_the_scores(self) -> None:
        position = Position(board([(7, 7, BLACK), (6, 6, WHITE)]))
        before = dict(position.totals)
        undo = position.play(7, 8, BLACK)
        self.assertNotEqual(position.totals[BLACK], before[BLACK])
        position.undo(undo)
        self.assertEqual(position.totals, before)
        self.assertEqual(position.grid[7][8], 0)
        self.assertEqual(len(position.stones), 2)

    def test_joining_two_runs_scores_the_whole_line(self) -> None:
        # H8 I8 _ J8 K8: playing the gap makes one run of five.
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 10, BLACK), (7, 11, BLACK)]))
        gain = position.gain_of(7, 9, BLACK)
        self.assertGreaterEqual(gain, WIN_SCORE // 2)

    def test_detects_a_five_through_the_new_stone(self) -> None:
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 10, BLACK), (7, 11, BLACK)]))
        self.assertTrue(position.wins_with(7, 9, BLACK))
        self.assertFalse(position.wins_with(6, 6, BLACK))

    def test_refuses_a_cell_that_holds_a_stone(self) -> None:
        position = Position(board([(7, 7, BLACK)]))
        with self.assertRaises(ValueError):
            position.play(7, 7, WHITE)

    def test_offers_only_empty_cells_next_to_a_stone(self) -> None:
        position = Position(board([(7, 7, BLACK)]))
        cells = position.neighbours()
        self.assertNotIn((7, 7), cells)
        self.assertIn((7, 8), cells)
        self.assertNotIn((0, 0), cells)

    def test_puts_the_block_of_a_four_first(self) -> None:
        # White has H8 I8 J8 K8; a black stone at L8 or G8 is the only defence.
        position = Position(
            board([(7, 7, WHITE), (7, 8, WHITE), (7, 9, WHITE), (7, 10, WHITE), (8, 8, BLACK)])
        )
        best = position.candidates(BLACK, 3)
        self.assertTrue({(7, 6), (7, 11)} & set(best), best)


class SearchTests(unittest.TestCase):
    def test_starts_in_the_center(self) -> None:
        move = Searcher(depth=2, width=10, time_budget=1.0).best_move(Position(empty()), BLACK)
        self.assertEqual(move, (7, 7))

    def test_takes_an_immediate_win(self) -> None:
        # Black H8 I8 J8 K8: L8 or G8 ends the game.
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK), (8, 8, WHITE)])
        )
        move = Searcher(depth=2, width=10, time_budget=1.0).best_move(position, BLACK)
        self.assertIn(coordinate_of(*move), ("G8", "L8"))

    def test_blocks_a_five_of_the_opponent(self) -> None:
        position = Position(
            board([(7, 7, WHITE), (7, 8, WHITE), (7, 9, WHITE), (7, 10, WHITE), (8, 8, BLACK)])
        )
        move = Searcher(depth=2, width=10, time_budget=1.0).best_move(position, BLACK)
        self.assertIn(coordinate_of(*move), ("G8", "L8"))

    def test_alpha_beta_finds_the_value_of_the_plain_search(self) -> None:
        # A quiet position, so no immediate win hides a difference.
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=4, width=8, time_budget=5.0)
        value, _ = searcher._root(position, BLACK, 4)
        plain = _plain_root(position.copy(), BLACK, 4, 8)
        self.assertEqual(value, plain)

    def test_alpha_beta_visits_fewer_nodes_than_the_plain_search(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=4, width=8, time_budget=5.0)
        searcher._root(position.copy(), BLACK, 4)
        pruned = searcher.nodes
        counted = [0]
        _plain_root(position.copy(), BLACK, 4, 8, counted)
        self.assertLess(pruned, counted[0])

    def test_keeps_the_time_budget(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=12, width=12, time_budget=0.2)
        move = searcher.best_move(position, BLACK)
        self.assertIsNotNone(move)
        self.assertEqual(position.grid[move[0]][move[1]], 0)

    def test_returns_nothing_on_a_full_board(self) -> None:
        grid = [[BLACK if (row + col) % 2 == 0 else WHITE for col in range(15)] for row in range(15)]
        self.assertIsNone(Searcher(depth=2, width=4, time_budget=0.5).best_move(Position(grid), BLACK))


def _plain_negamax(
    position: Position,
    depth: int,
    player: int,
    width: int,
    total_depth: int,
    counted: list[int] | None = None,
) -> int:
    """A negamax without pruning, to compare the pruned one with."""
    if counted is not None:
        counted[0] += 1
    if depth <= 0:
        return position.evaluate(player)
    moves = position.candidates(player, width)
    if not moves:
        return position.evaluate(player)
    best = -INFINITY
    for row, col in moves:
        undo = position.play(row, col, player)
        if undo.win:
            value = WIN_SCORE - (total_depth - depth)
        else:
            value = -_plain_negamax(position, depth - 1, other(player), width, total_depth, counted)
        position.undo(undo)
        if value > best:
            best = value
    return best


def _plain_root(
    position: Position,
    player: int,
    depth: int,
    width: int,
    counted: list[int] | None = None,
) -> int:
    """The value of the best move, without alpha-beta pruning."""
    best = -INFINITY
    for row, col in position.candidates(player, width):
        undo = position.play(row, col, player)
        if undo.win:
            value = WIN_SCORE
        else:
            value = -_plain_negamax(position, depth - 1, other(player), width, depth, counted)
        position.undo(undo)
        if value > best:
            best = value
    return best


class EngineTests(unittest.TestCase):
    def test_answers_a_free_cell(self) -> None:
        engine = ClassicalEngine(depth=2, width=8, time_budget=0.5)
        move = engine.choose_move(board([(7, 7, BLACK), (7, 8, WHITE)]), [], BLACK)
        self.assertEqual(board([(7, 7, BLACK), (7, 8, WHITE)])[move[0]][move[1]], 0)

    def test_the_game_settings_win_over_the_worker_ones(self) -> None:
        engine = ClassicalEngine(depth=2, width=8, time_budget=0.5)
        move = engine.choose_move(empty(), [], BLACK, None, {"depth": 4, "width": 4})
        self.assertEqual(move, (7, 7))

    def test_reports_a_full_board(self) -> None:
        grid = [[BLACK if (row + col) % 2 == 0 else WHITE for col in range(15)] for row in range(15)]
        with self.assertRaises(EngineError):
            ClassicalEngine().choose_move(grid, [], BLACK)


class SettingsTests(unittest.TestCase):
    def test_reads_the_search_parameters(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_ENGINE_TOKEN": "token",
                "CLASSICAL_DEPTH": "6",
                "CLASSICAL_WIDTH": "12",
                "CLASSICAL_TIME_BUDGET": "0.8",
            }
        )
        self.assertEqual(settings.depth, 6)
        self.assertEqual(settings.width, 12)
        self.assertEqual(settings.time_budget, 0.8)
        self.assertEqual(settings.connection.engine_id, "classical")
        self.assertEqual(settings.connection.worker_id, "classical-worker-01")

    def test_reports_a_parameter_that_is_not_a_number(self) -> None:
        with self.assertRaises(ConfigError):
            Settings.from_env({"MYGO_ENGINE_TOKEN": "token", "CLASSICAL_DEPTH": "deep"})


if __name__ == "__main__":
    unittest.main()
