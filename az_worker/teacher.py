from __future__ import annotations

import math

import numpy as np

from classical_worker.position import Position
from classical_worker.search import Searcher
from classical_worker.threats import ordered_moves

from .board import BLACK, BOARD_SIZE, WHITE, Board, other
from .features import encode, legal_mask
from .mcts import MCTS, row_col_of
from .selfplay import Example


VALUE_OUTCOME = "outcome"
VALUE_EVAL = "eval"
VALUE_SEARCH = "search"


def teacher_policy(ranked: list[tuple[int, int]], decay: float) -> np.ndarray:
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    for index, (row, col) in enumerate(ranked):
        policy[row * BOARD_SIZE + col] = math.exp(-index / decay)
    return policy


def collect_one_game(
    depth: int,
    width: int,
    time_budget: float,
    teacher_moves: int,
    decay: float,
    score_scale: float,
    value_target: str,
) -> list[Example]:
    searcher = Searcher(depth=depth, width=width, time_budget=time_budget)
    examples: list[Example] = []
    board = Board()
    position = Position()
    player = BLACK
    last_move: tuple[int, int] | None = None
    winner: int | None = None
    while not board.is_full():
        move = searcher.best_move(position, player)
        if move is None:
            break

        ranked = ordered_moves(position, player, teacher_moves)
        if ranked and ranked[0] != move:
            ranked = [move] + [cell for cell in ranked if cell != move]
        examples.append(
            Example(
                planes=encode(board, player, last_move),
                policy=teacher_policy(ranked, decay),
                legal=legal_mask(board, 1),
                value=(
                    math.tanh(position.evaluate(player) / score_scale)
                    if value_target == VALUE_EVAL
                    else 0.0
                ),
                teacher_cell=move[0] * BOARD_SIZE + move[1],
            )
        )
        row, col = move
        last_move = (row, col)
        if board.play(row, col, player):
            winner = player
            break
        position.play(row, col, player)
        player = other(player)
    if value_target == VALUE_OUTCOME:
        for index, example in enumerate(examples):
            mover = BLACK if index % 2 == 0 else WHITE
            if winner is None:
                example.value = 0.0
            else:
                example.value = 1.0 if winner == mover else -1.0
    return examples


def collect_az_game(
    model,
    device,
    simulations: int,
    time_budget: float | None,
    candidate_width: int,
    value_target: str = VALUE_SEARCH,
    seed: int | None = None,
) -> list[Example]:
    if value_target not in (VALUE_SEARCH, VALUE_OUTCOME):
        raise ValueError(f"AZ 老师的价值目标只能是 search 或 outcome: {value_target!r}")
    searcher = MCTS(
        model,
        device,
        candidate_width=candidate_width,
        time_budget=time_budget,
        seed=seed,
    )
    examples: list[Example] = []
    board = Board()
    player = BLACK
    last_move: tuple[int, int] | None = None
    winner: int | None = None
    while not board.is_full():
        result = searcher.search(
            board,
            player,
            simulations=max(1, simulations),
            add_noise=True,
            last_move=last_move,
        )
        move = result.best_move()
        if move is None:
            break
        row, col = row_col_of(move)
        examples.append(
            Example(
                planes=encode(board, player, last_move),
                policy=result.policy,
                legal=legal_mask(board, 1),
                value=result.root_value if value_target == VALUE_SEARCH else 0.0,
                teacher_cell=move,
            )
        )
        last_move = (row, col)
        if board.play(row, col, player):
            winner = player
            break
        player = other(player)
    if value_target == VALUE_OUTCOME:
        for index, example in enumerate(examples):
            mover = BLACK if index % 2 == 0 else WHITE
            if winner is None:
                example.value = 0.0
            else:
                example.value = 1.0 if winner == mover else -1.0
    return examples
