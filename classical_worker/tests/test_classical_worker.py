from __future__ import annotations

import random
import unittest

from classical_worker.config import ConfigError, Settings
from classical_worker.engine import ClassicalEngine, EngineError
from classical_worker.patterns import WINDOW_SCORES, classify as classify_window, decode
from classical_worker.position import BLACK, WHITE, Position, coordinate_of, other
from classical_worker.search import INFINITY, Searcher
from classical_worker.shapes import CLOSED_FOUR, OPEN_THREE, board_threats, threat_profile
from classical_worker.threats import VCF, VCT, classify, forced_moves, ordered_moves
from classical_worker.tt import MATE_SCORE, EXACT, LOWER, SIDE_KEY, TranspositionTable, from_table, to_table


def board(stones: list[tuple[int, int, int]]) -> list[list[int]]:
    grid = [[0] * 15 for _ in range(15)]
    for row, col, player in stones:
        grid[row][col] = player
    return grid


def empty() -> list[list[int]]:
    return [[0] * 15 for _ in range(15)]


class PatternTests(unittest.TestCase):
    def test_reads_the_shapes_from_the_design(self) -> None:
        cases = {
            (0, 1, 1, 1, 1, 1): "five",
            (0, 1, 1, 1, 1, 0): "open_four",
            (0, 1, 1, 1, 1, 2): "four",
            (2, 1, 1, 1, 1, 0): "four",
            (0, 1, 1, 0, 1, 1): "four",
            (1, 0, 1, 1, 1, 0): "four",
            (1, 1, 1, 0, 1, 0): "four",
            (0, 1, 1, 1, 0, 0): "open_three",
            (0, 0, 1, 1, 1, 0): "open_three",
            (0, 1, 0, 1, 1, 0): "open_three",
            (0, 1, 1, 0, 1, 0): "open_three",
            (0, 0, 1, 1, 1, 2): "closed_three",
            (2, 1, 1, 1, 0, 0): "closed_three",
            (0, 0, 1, 1, 0, 0): "open_two",
            (0, 1, 0, 1, 0, 0): "open_two",
            (0, 0, 1, 1, 2, 0): "closed_two",
            (0, 2, 1, 1, 0, 0): "closed_two",
            (0, 0, 1, 0, 0, 0): "one",
        }
        for cells, name in cases.items():
            self.assertEqual(classify_window(cells)[0], name, cells)

    def test_scores_the_two_colours_symmetrically(self) -> None:
        for code in range(3**6):
            cells = decode(code)
            swapped = [0 if value == 0 else 3 - value for value in cells]
            code_swapped = 0
            for index, value in enumerate(swapped):
                code_swapped += value * (3**index)
            self.assertEqual(WINDOW_SCORES[code][0], WINDOW_SCORES[code_swapped][1])

    def test_orders_the_shapes(self) -> None:
        five = classify_window((1, 1, 1, 1, 1, 0))[1]
        open_four = classify_window((0, 1, 1, 1, 1, 0))[1]
        four = classify_window((0, 1, 1, 1, 1, 2))[1]
        open_three = classify_window((0, 1, 1, 1, 0, 0))[1]
        self.assertGreater(five, open_four)
        self.assertGreater(open_four, four)
        self.assertGreater(four, open_three)


class PositionTests(unittest.TestCase):
    def test_scores_an_open_run_above_a_closed_one(self) -> None:
        open_run = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)]))
        closed_run = Position(board([(0, 0, BLACK), (0, 1, BLACK), (0, 2, BLACK)]))
        self.assertGreater(open_run.evaluate(BLACK), closed_run.evaluate(BLACK))

    def test_scores_the_players_from_their_own_side(self) -> None:
        position = Position(board([(7, 7, BLACK)]))
        self.assertEqual(position.evaluate(BLACK), -position.evaluate(WHITE))

    def test_play_and_undo_restore_the_scores_and_the_hash(self) -> None:
        position = Position(board([(7, 7, BLACK), (6, 6, WHITE)]))
        before = (dict(position.totals), position.hash, len(position.stones))
        undo = position.play(7, 8, BLACK)
        self.assertNotEqual(position.totals[BLACK], before[0][BLACK])
        position.undo(undo)
        self.assertEqual((position.totals, position.hash, len(position.stones)), before)
        self.assertEqual(position.grid[7][8], 0)

    def test_incremental_scores_match_a_full_recount(self) -> None:
        rng = random.Random(11)
        position = Position()
        player = BLACK
        for _ in range(300):
            cells = position.neighbours()
            if not cells:
                break
            undo = position.play(*rng.choice(cells), player)
            totals = dict(position.totals)
            hashed = position.hash
            position.recount()
            self.assertEqual(position.totals, totals)
            self.assertEqual(position.hash, hashed)
            player = other(player)
            if rng.random() < 0.4:
                position.undo(undo)
                player = other(player)

    def test_offers_only_empty_cells_near_a_stone(self) -> None:
        position = Position(board([(7, 7, BLACK)]))
        cells = position.neighbours()
        self.assertNotIn((7, 7), cells)
        self.assertIn((7, 8), cells)
        self.assertNotIn((7, 9), cells)
        self.assertNotIn((0, 0), cells)
        self.assertIn((7, 9), position.neighbours(radius=2))

    def test_puts_the_block_of_a_four_first(self) -> None:
        position = Position(
            board([(7, 7, WHITE), (7, 8, WHITE), (7, 9, WHITE), (7, 10, WHITE), (8, 8, BLACK)])
        )
        best = ordered_moves(position, BLACK, 4)
        self.assertTrue({(7, 6), (7, 11)} & set(best), best)


class ThreatTests(unittest.TestCase):
    def test_an_open_three_has_two_ways_to_an_open_four(self) -> None:
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK)]))
        threat = classify(position, 7, 10, BLACK)
        self.assertFalse(threat.five)
        self.assertGreaterEqual(threat.five_points, 2)
        self.assertGreaterEqual(threat.open_fours, 1)

    def test_a_blocked_three_is_not_an_open_three(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 6, WHITE)])
        )
        threat = classify(position, 7, 10, BLACK)
        self.assertFalse(threat.five)
        self.assertEqual(threat.five_points, 1)
        self.assertEqual(threat.open_fours, 0)

    def test_a_gap_four_makes_five(self) -> None:
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 10, BLACK), (7, 11, BLACK)]))
        threat = classify(position, 7, 9, BLACK)
        self.assertTrue(threat.five)

    def test_the_threat_table_agrees_with_wins_with(self) -> None:
        rng = random.Random(5)
        position = Position()
        player = BLACK
        for _ in range(40):
            cells = position.neighbours()
            if not cells:
                break
            for row, col in cells:
                self.assertEqual(
                    classify(position, row, col, BLACK).five,
                    position.wins_with(row, col, BLACK),
                    (row, col),
                )
                self.assertEqual(
                    classify(position, row, col, WHITE).five,
                    position.wins_with(row, col, WHITE),
                    (row, col),
                )
            position.play(*rng.choice(cells), player)
            player = other(player)

    def test_forced_moves_take_the_win_and_block_the_five(self) -> None:
        winning = Position(board([(7, 7, BLACK), (7, 8, BLACK), (7, 9, BLACK), (7, 10, BLACK)]))
        self.assertIn((7, 6), forced_moves(winning, BLACK, 4))
        blocking = Position(board([(7, 7, WHITE), (7, 8, WHITE), (7, 9, WHITE), (7, 10, WHITE)]))
        self.assertEqual(forced_moves(blocking, BLACK, 4), [(7, 6), (7, 11)])


class ThreatRankTests(unittest.TestCase):
    def test_one_open_three_counts_as_one_line(self) -> None:
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK)]))
        threat = classify(position, 7, 9, BLACK)
        self.assertEqual(threat.open_threes, 1)
        self.assertEqual(threat.open_fours, 0)

    def test_a_double_three_counts_two_lines(self) -> None:
        position = Position(
            board([(7, 5, BLACK), (7, 6, BLACK), (5, 7, BLACK), (6, 7, BLACK)])
        )
        threat = classify(position, 7, 7, BLACK)
        self.assertEqual(threat.open_threes, 2)
        self.assertEqual(threat.five_points, 0)

    def test_a_four_three_outranks_a_double_three(self) -> None:
        four_three = classify(
            Position(
                board(
                    [
                        (7, 4, BLACK),
                        (7, 5, BLACK),
                        (7, 6, BLACK),
                        (7, 3, WHITE),
                        (5, 7, BLACK),
                        (6, 7, BLACK),
                    ]
                )
            ),
            7,
            7,
            BLACK,
        )
        double_three = classify(
            Position(board([(7, 5, BLACK), (7, 6, BLACK), (5, 7, BLACK), (6, 7, BLACK)])),
            7,
            7,
            BLACK,
        )
        self.assertEqual((four_three.five_points, four_three.open_threes), (1, 1))
        self.assertGreater(four_three.rank, double_three.rank)

    def test_an_open_four_outranks_every_combination(self) -> None:
        open_four = classify(
            Position(board([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])), 7, 7, BLACK
        )
        four_three = classify(
            Position(
                board(
                    [
                        (7, 4, BLACK),
                        (7, 5, BLACK),
                        (7, 6, BLACK),
                        (7, 3, WHITE),
                        (5, 7, BLACK),
                        (6, 7, BLACK),
                    ]
                )
            ),
            7,
            7,
            BLACK,
        )
        self.assertEqual((open_four.five_points, open_four.open_fours), (2, 1))
        self.assertGreater(open_four.rank, four_three.rank)


class ShapeProfileTests(unittest.TestCase):
    def test_reads_an_open_four_with_two_five_points(self) -> None:
        profile = threat_profile(
            Position(board([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])),
            BLACK,
        )
        self.assertEqual(len(profile.open_fours), 1)
        threat = profile.open_fours[0]
        self.assertEqual(threat.stones, frozenset({(7, 4), (7, 5), (7, 6), (7, 7)}))
        self.assertEqual(threat.points, ((7, 3), (7, 8)))
        self.assertEqual(threat.blocks, ())
        self.assertEqual(profile.five_points, frozenset({(7, 3), (7, 8)}))

    def test_reads_a_closed_four_with_one_five_point(self) -> None:
        profile = threat_profile(
            Position(
                board([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (7, 3, WHITE)])
            ),
            BLACK,
        )
        self.assertEqual(len(profile.closed_fours), 1)
        threat = profile.closed_fours[0]
        self.assertEqual(threat.kind, CLOSED_FOUR)
        self.assertEqual(threat.points, ((7, 8),))
        self.assertEqual(threat.blocks, ((7, 8),))

    def test_reads_the_open_three_extension_points(self) -> None:
        profile = threat_profile(
            Position(board([(7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])), BLACK
        )
        self.assertEqual(len(profile.open_threes), 1)
        threat = profile.open_threes[0]
        self.assertEqual(threat.kind, OPEN_THREE)
        self.assertEqual(threat.points, ((7, 4), (7, 8)))
        self.assertEqual(threat.blocks, ((7, 4), (7, 8)))
        self.assertEqual([extension for extension, _ in threat.extensions], [(7, 4), (7, 8)])

    def test_a_broken_three_has_one_extension(self) -> None:
        profile = threat_profile(
            Position(board([(7, 1, BLACK), (7, 3, BLACK), (7, 4, BLACK)])), BLACK
        )
        threat = profile.open_threes[0]
        self.assertEqual(threat.points, ((7, 2),))
        self.assertEqual(threat.blocks, ((7, 0), (7, 2), (7, 5)))

    def test_a_blocked_three_is_not_a_threat(self) -> None:
        profile = threat_profile(
            Position(board([(7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (7, 4, WHITE)])),
            BLACK,
        )
        self.assertEqual(profile.threats, ())
        self.assertEqual(profile.block_points, frozenset())

    def test_reports_the_threats_of_both_sides(self) -> None:
        black, white = board_threats(
            Position(
                board(
                    [
                        (7, 4, BLACK),
                        (7, 5, BLACK),
                        (7, 6, BLACK),
                        (3, 4, WHITE),
                        (3, 5, WHITE),
                        (3, 6, WHITE),
                    ]
                )
            )
        )
        self.assertEqual(black.player, BLACK)
        self.assertEqual(white.player, WHITE)
        self.assertEqual(set(black.block_points), {(7, 3), (7, 7)})
        self.assertEqual(set(white.block_points), {(3, 3), (3, 7)})


class CombinationTests(unittest.TestCase):
    def test_extends_an_open_three_to_a_winning_open_four(self) -> None:
        position = Position(board([(7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)]))
        move = VCT().best_move(position, BLACK, float("inf"))
        self.assertIn(move, {(7, 4), (7, 8)})

    def test_finds_the_four_three_win(self) -> None:
        position = Position(
            board(
                [
                    (7, 4, BLACK),
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (7, 3, WHITE),
                    (5, 7, BLACK),
                    (6, 7, BLACK),
                    (0, 0, WHITE),
                    (0, 2, WHITE),
                ]
            )
        )
        self.assertEqual(VCT().best_move(position, BLACK, float("inf")), (7, 7))

    def test_finds_the_double_three_win_through_a_counter_four(self) -> None:
        position = Position(
            board(
                [
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (5, 7, BLACK),
                    (6, 7, BLACK),
                    (0, 0, WHITE),
                    (0, 1, WHITE),
                    (0, 2, WHITE),
                ]
            )
        )
        self.assertEqual(VCT().best_move(position, BLACK, float("inf")), (7, 7))

    def test_refuses_a_double_three_that_loses_to_an_open_four(self) -> None:
        position = Position(
            board(
                [
                    (7, 5, BLACK),
                    (7, 6, BLACK),
                    (5, 7, BLACK),
                    (6, 7, BLACK),
                    (3, 3, WHITE),
                    (3, 4, WHITE),
                    (3, 5, WHITE),
                ]
            )
        )
        self.assertIsNone(VCT().best_move(position, BLACK, float("inf")))


class SearchTests(unittest.TestCase):
    def test_starts_in_the_center(self) -> None:
        move = Searcher(depth=2, width=10, time_budget=1.0).best_move(Position(empty()), BLACK)
        self.assertEqual(move, (7, 7))

    def test_takes_an_immediate_win(self) -> None:
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
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=4, width=8, time_budget=10.0, vcf_budget=0, vct_budget=0)
        value, _ = searcher._root(position.copy(), BLACK, 4)
        plain = _plain_root(position.copy(), BLACK, 4, 8)
        self.assertEqual(value, plain)

    def test_alpha_beta_visits_fewer_nodes_than_the_plain_search(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=4, width=8, time_budget=10.0, vcf_budget=0, vct_budget=0)
        searcher._root(position.copy(), BLACK, 4)
        pruned = searcher.nodes
        counted = [0]
        _plain_root(position.copy(), BLACK, 4, 8, counted)
        self.assertLess(pruned, counted[0])

    def test_a_repeated_search_uses_the_table(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=4, width=8, time_budget=10.0, vcf_budget=0, vct_budget=0)
        first, _ = searcher._root(position.copy(), BLACK, 4)
        second, _ = searcher._root(position.copy(), BLACK, 4)
        self.assertEqual(first, second)
        self.assertGreater(searcher.table.hits, 0)

    def test_keeps_the_time_budget(self) -> None:
        position = Position(
            board([(7, 7, BLACK), (7, 8, BLACK), (6, 6, WHITE), (8, 8, WHITE), (6, 8, BLACK)])
        )
        searcher = Searcher(depth=12, width=12, time_budget=0.2)
        move = searcher.best_move(position, BLACK)
        self.assertIsNotNone(move)
        self.assertEqual(position.grid[move[0]][move[1]], 0)

    def test_finds_a_vcf_win(self) -> None:
        position = Position(
            board(
                [
                    (7, 7, BLACK),
                    (7, 8, BLACK),
                    (7, 9, BLACK),
                    (8, 7, BLACK),
                    (8, 8, BLACK),
                    (9, 9, WHITE),
                ]
            )
        )
        move = VCF().best_move(position, BLACK, float("inf"))
        self.assertIsNotNone(move)

    def test_vcf_refuses_a_line_where_the_opponent_wins_first(self) -> None:
        position = Position(
            board(
                [
                    (7, 7, BLACK),
                    (7, 8, BLACK),
                    (7, 9, BLACK),
                    (0, 0, WHITE),
                    (0, 1, WHITE),
                    (0, 2, WHITE),
                    (0, 3, WHITE),
                ]
            )
        )
        self.assertIsNone(VCF().best_move(position, BLACK, float("inf")))

    def test_vct_does_not_claim_an_open_three_as_a_win(self) -> None:
        position = Position(board([(7, 7, BLACK), (7, 8, BLACK)]))
        self.assertIsNone(VCT(max_depth=2).best_move(position, BLACK, float("inf")))

    def test_returns_nothing_on_a_full_board(self) -> None:
        grid = [[BLACK if (row + col) % 2 == 0 else WHITE for col in range(15)] for row in range(15)]
        self.assertIsNone(Searcher(depth=2, width=4, time_budget=0.5).best_move(Position(grid), BLACK))


def _plain_negamax(
    position: Position,
    depth: int,
    player: int,
    width: int,
    ply: int,
    counted: list[int] | None = None,
) -> int:
    if counted is not None:
        counted[0] += 1
    if depth <= 0:
        return position.evaluate(player)
    if position.is_full():
        return 0
    moves = forced_moves(position, player, width)
    if not moves:
        return 0
    best = -INFINITY
    for move in moves:
        undo = position.play(*move, player)
        if undo.win:
            value = MATE_SCORE - (ply + 1)
        else:
            value = -_plain_negamax(position, depth - 1, other(player), width, ply + 1, counted)
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
    best = -INFINITY
    for move in forced_moves(position, player, width * 2):
        undo = position.play(*move, player)
        if undo.win:
            value = MATE_SCORE - 1
        else:
            value = -_plain_negamax(position, depth - 1, other(player), width, 1, counted)
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


class TableTests(unittest.TestCase):
    def test_normalizes_a_mate_score(self) -> None:
        value = MATE_SCORE - 5
        self.assertEqual(from_table(to_table(value, 3), 3), value)
        self.assertEqual(from_table(to_table(-value, 3), 3), -value)

    def test_keeps_a_deeper_entry(self) -> None:
        table = TranspositionTable(1024)
        table.store(7, 4, 10, EXACT, (1, 2))
        table.store(7, 2, 20, LOWER, (3, 4))
        self.assertEqual(table.probe(7).depth, 4)
        self.assertEqual(table.probe(7).value, 10)

    def test_the_side_to_move_changes_the_key(self) -> None:
        self.assertNotEqual(SIDE_KEY[BLACK], SIDE_KEY[WHITE])


class SettingsTests(unittest.TestCase):
    def test_reads_the_search_parameters(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_ENGINE_TOKEN": "token",
                "CLASSICAL_DEPTH": "6",
                "CLASSICAL_WIDTH": "12",
                "CLASSICAL_TIME_BUDGET": "0.8",
                "CLASSICAL_VCF_BUDGET": "0.1",
                "CLASSICAL_VCT_BUDGET": "0.2",
                "CLASSICAL_TT_ENTRIES": "4096",
            }
        )
        self.assertEqual(settings.depth, 6)
        self.assertEqual(settings.width, 12)
        self.assertEqual(settings.time_budget, 0.8)
        self.assertEqual(settings.vcf_budget, 0.1)
        self.assertEqual(settings.vct_budget, 0.2)
        self.assertEqual(settings.tt_entries, 4096)
        self.assertEqual(settings.connection.engine_id, "classical")
        self.assertEqual(settings.connection.worker_id, "classical-worker-01")

    def test_reports_a_parameter_that_is_not_a_number(self) -> None:
        with self.assertRaises(ConfigError):
            Settings.from_env({"MYGO_ENGINE_TOKEN": "token", "CLASSICAL_DEPTH": "deep"})


if __name__ == "__main__":
    unittest.main()
