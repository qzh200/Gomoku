"""Shared, exact one/two-ply tactics for freestyle Gomoku (no forbidden moves).

Only proven forcing moves prune the tree. Creating a three is a preference,
not a proof of a win; longer combinations remain the job of MCTS / VCF / VCT.
"""
from __future__ import annotations

import numpy as np

from .board import BOARD_SIZE
from .threats import (
    STRIDE, bits, board_masks, five_mask, open_four_mask, open_three_mask,
)


def immediate_moves(board, player: int) -> list[tuple[int, int]]:
    own, opponent, empty = board_masks(board, player)
    targets = five_mask(own, empty) or five_mask(opponent, empty)
    return [divmod(index, STRIDE) for index in bits(targets)]


def search_moves(board, player: int, moves, policy: np.ndarray):
    """Return tactical candidates and priors, including moves outside radius.

    Against a live three, keep every move preventing an open four, as well as
    counter-checks: forcing the opponent to defend can be better than blocking.
    If no defense exists, keep the candidates rather than inventing a safe move.
    """
    own, opponent, empty = board_masks(board, player)
    wins = five_mask(own, empty)
    checks = five_mask(opponent, empty)
    fours = open_four_mask(own, empty)
    enemy_fours = open_four_mask(opponent, empty)
    forced = wins or checks or fours
    if forced:
        moves = [divmod(index, STRIDE) for index in bits(forced)]
    elif enemy_fours:
        defenses = []
        for index in bits(empty):
            bit = 1 << index
            remaining = empty & ~bit
            if (five_mask(own | bit, remaining)
                    or not open_four_mask(opponent, remaining)):
                defenses.append(divmod(index, STRIDE))
        if defenses:
            moves = defenses

    # A small tactical prior floor prevents a bad network from hiding useful
    # three-building moves. It never turns a heuristic into a forced answer.
    threes = open_three_mask(own, empty)
    enemy_threes = open_three_mask(opponent, empty)
    priors = np.array([policy[row * BOARD_SIZE + col] for row, col in moves], dtype=np.float64)
    weights = np.array([
        1.0 + 3.0 * bool(threes & (1 << (row * STRIDE + col)))
        + bool(enemy_threes & (1 << (row * STRIDE + col)))
        for row, col in moves
    ])
    if len(priors):
        priors = 0.9 * priors / max(float(priors.sum()), 1e-12) + 0.1 * weights / weights.sum()
        priors /= priors.sum()
    return moves, priors, wins, checks, fours
