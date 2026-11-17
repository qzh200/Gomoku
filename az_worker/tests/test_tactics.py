from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from az_worker.board import BLACK, WHITE, Board
from az_worker.mcts import MCTS, Node, cell_of
from az_worker.selfplay import SelfPlay
from az_worker.threats import board_masks, open_four_mask
from az_worker.train import build_parser
from az_worker.vcf import PlaySolver, Solver, VCT
from az_worker.tests.test_az_worker import board_of, permute_board, permute_cell


def misleading_network(planes, masks):
    # Put essentially all probability at an irrelevant corner and claim a win.
    priors = masks.astype(np.float32) * 1e-12
    priors[:, 0] += masks[:, 0]
    priors /= priors.sum(1, keepdims=True)
    return priors, np.full(len(planes), -0.99, dtype=np.float32)


class TacticalTests(unittest.TestCase):
    def search(self, board, player=BLACK, simulations=16, **kwargs):
        return MCTS(None, torch.device('cpu'), evaluator=misleading_network,
                    batch_size=8, seed=7, **kwargs).search(
                        board, player, simulations=simulations, add_noise=True)

    def test_immediate_win_and_block_survive_bad_priors_and_all_symmetries(self):
        # Both a straight, closed four and an internal-gap four.
        for cols, target in [([4, 5, 6, 7], (7, 8)), ([4, 5, 7, 8], (7, 6))]:
            for attacker in (BLACK, WHITE):
                board = Board(board_of([(7, c, attacker) for c in cols]
                                       + [(7, 3, 3 - attacker)]))
                for symmetry in range(8):
                    position = permute_board(board, symmetry)
                    expected = cell_of(*permute_cell(target, symmetry))
                    for radius in (0, 1, 2):
                        with self.subTest(cols=cols, attacker=attacker,
                                          symmetry=symmetry, radius=radius):
                            result = self.search(position, radius=radius)
                            self.assertEqual(result.best_move(), expected)
                            self.assertAlmostEqual(float(result.policy[expected]), 1.0)

    def test_winning_is_more_urgent_than_blocking(self):
        board = Board(board_of([(7, c, BLACK) for c in range(4, 8)]
                               + [(9, c, WHITE) for c in range(4, 8)]))
        result = self.search(board)
        self.assertIn(result.best_move(), {cell_of(7, 3), cell_of(7, 8)})
        self.assertTrue(all(board.wins_with(*divmod(c, 15), BLACK)
                            for c in result.visits))

    def test_a_double_immediate_threat_has_proven_losing_value(self):
        board = Board(board_of([(7, c, WHITE) for c in range(4, 8)]))
        result = self.search(board)
        self.assertAlmostEqual(result.root_value, -1.0)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0)

    def test_open_three_creates_an_open_four_despite_bad_priors(self):
        board = Board(board_of([(7, c, BLACK) for c in (5, 6, 7)]))
        for symmetry in range(8):
            position = permute_board(board, symmetry)
            expected = {cell_of(*permute_cell(cell, symmetry))
                        for cell in [(7, 4), (7, 8)]}
            result = self.search(position)
            self.assertIn(result.best_move(), expected)
            self.assertEqual(result.root_value, 1.0)

    def test_straight_and_broken_open_threes_are_defended(self):
        for cols in [(5, 6, 7), (5, 6, 8)]:
            board = Board(board_of([(7, c, WHITE) for c in cols]))
            for symmetry in range(8):
                position = permute_board(board, symmetry)
                result = self.search(position)
                for move in result.visits:
                    reply = position.copy()
                    reply.play(*divmod(move, 15), BLACK)
                    own, opponent, empty = board_masks(reply, BLACK)
                    self.assertFalse(open_four_mask(opponent, empty))

    def test_defense_keeps_a_counter_check(self):
        board = Board(board_of([(7, c, WHITE) for c in (5, 6, 7)]
                               + [(3, c, BLACK) for c in (4, 5, 6)]
                               + [(3, 3, WHITE)]))
        root = Node(board, BLACK)
        MCTS(None, 'cpu')._expand(root, np.full(225, 1 / 225))
        # This creates a closed four and obliges White to answer it.
        self.assertIn(cell_of(3, 7), root.children)
        self.assertNotIn(cell_of(0, 0), root.children)

    def test_live_four_creation_wins_before_defending_an_open_three(self):
        board = Board(board_of([(7, c, WHITE) for c in (5, 6, 7)]
                               + [(3, c, BLACK) for c in (4, 5, 6)]))
        result = self.search(board)
        self.assertIn(result.best_move(), {cell_of(3, 3), cell_of(3, 7)})

    def test_a_closed_three_is_not_mistaken_for_an_open_three(self):
        board = Board(board_of([(7, c, WHITE) for c in (5, 6, 7)]
                               + [(7, 4, BLACK)]))
        root = Node(board, BLACK)
        MCTS(None, 'cpu')._expand(root, np.full(225, 1 / 225))
        self.assertIn(cell_of(0, 0), root.children)

    def test_two_stones_do_not_force_a_single_three_building_move(self):
        board = Board(board_of([(7, 6, BLACK), (7, 7, BLACK)]))
        root = Node(board, BLACK)
        MCTS(None, 'cpu')._expand(root, np.full(225, 1 / 225))
        self.assertEqual(len(root.children), 223)
        self.assertGreater(root.children[cell_of(7, 8)].prior,
                           root.children[cell_of(0, 0)].prior)

    def test_edges_do_not_wrap_and_edge_fours_are_blocked(self):
        board = Board(board_of([(0, c, WHITE) for c in range(4)]))
        self.assertEqual(self.search(board).best_move(), cell_of(0, 4))
        board = Board(board_of([(r, 14, WHITE) for r in range(4)]))
        self.assertEqual(self.search(board).best_move(), cell_of(4, 14))

    def test_solver_blocks_even_with_both_search_budgets_disabled(self):
        board = Board(board_of([(7, c, WHITE) for c in range(4, 8)]
                               + [(7, 3, BLACK)]))
        self.assertEqual(PlaySolver(0, 0).move(board, BLACK), (7, 8))

    def test_nested_nodes_apply_the_same_defense(self):
        # Expanding a non-root node must filter threats just like the root.
        board = Board(board_of([(7, c, BLACK) for c in (4, 5, 6)]
                               + [(7, 3, WHITE)]))
        parent = Node(board, BLACK)
        child = Node(None, WHITE, parent=parent, move=(7, 7))
        MCTS(None, 'cpu')._expand(child, np.full(225, 1 / 225))
        self.assertEqual(set(child.children), {cell_of(7, 8)})

    def test_selfplay_never_learns_to_ignore_an_immediate_single_threat(self):
        game = SelfPlay(None, torch.device('cpu'), simulations=8, batch_size=4,
                        seed=11, evaluator=misleading_network).play_game()
        checks = 0
        for example in game.examples:
            own_five = example.planes[3].reshape(-1)
            enemy_five = example.planes[4].reshape(-1)
            if enemy_five.sum() == 1 and own_five.sum() == 0:
                checks += 1
                self.assertAlmostEqual(float((example.policy * enemy_five).sum()), 1.0)
        self.assertGreater(checks, 0)

    def test_double_three_combination_is_found_by_vct(self):
        import time
        board = Board(board_of([(7, 6, BLACK), (7, 8, BLACK),
                                (6, 7, BLACK), (8, 7, BLACK)]))
        move = Solver(max_depth=8).winning_move(board, BLACK, time.monotonic() + 2, VCT)
        self.assertEqual(move, (7, 7))

    def test_vct_checks_outer_endpoint_defenses_of_a_broken_three(self):
        board = Board(board_of([(7, c, BLACK) for c in (5, 6, 8)]))
        own, opponent, empty = board_masks(board, BLACK)
        endpoint = 1 << (7 * 20 + 4)
        # A continuation that refutes this endpoint must prevent the solver
        # from declaring victory based solely on blocking the internal gap.
        with patch.object(Solver, '_attack', side_effect=lambda a, b, e, d, m: not bool(b & endpoint)):
            self.assertFalse(Solver()._defend(own, opponent, empty, 4, VCT))

    def test_vct_considers_a_defenders_counter_check_on_another_line(self):
        board = Board(board_of([(7, c, BLACK) for c in (5, 6, 7)]
                               + [(3, c, WHITE) for c in (4, 5, 6)]
                               + [(3, 3, BLACK)]))
        own, opponent, empty = board_masks(board, BLACK)
        counter = 1 << (3 * 20 + 7)
        with patch.object(Solver, '_attack', side_effect=lambda a, b, e, d, m: not bool(b & counter)):
            self.assertFalse(Solver()._defend(own, opponent, empty, 4, VCT))

    def test_tactics_do_not_change_the_board(self):
        board = Board(board_of([(7, c, WHITE) for c in (5, 6, 7)]))
        before = [row[:] for row in board.grid], board.stones[:]
        self.search(board)
        self.assertEqual((board.grid, board.stones), before)

    def test_training_defaults_favor_new_data_and_augmentation(self):
        args = build_parser().parse_args([])
        self.assertTrue(args.augment)
        self.assertFalse(build_parser().parse_args(['--no-augment']).augment)
        self.assertEqual(args.candidate_radius, 2)
        self.assertLessEqual(args.total_steps / args.iterations, 25)


if __name__ == '__main__':
    unittest.main()
