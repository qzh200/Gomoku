from __future__ import annotations

import math

from classical_worker.position import BLACK, Position, other
from classical_worker.search import Searcher
from classical_worker.threats import ordered_moves

from .board import BOARD_SIZE, Board
from .features import encode, legal_mask
from .selfplay import Example


VALUE_OUTCOME = "outcome"
VALUE_EVAL = "eval"


def teacher_policy(ranked: list[tuple[int, int]], decay: float) -> list[float]:
    policy = [0.0] * (BOARD_SIZE * BOARD_SIZE)
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
    """用传统引擎下一整局，把每一步的样本记下来。

    这个函数放在独立模块里，多进程收集时才能被子进程按名字导入
    （Windows 的 spawn 没法重新导入 __main__）。
    """
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
