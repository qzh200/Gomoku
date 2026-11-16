from __future__ import annotations

import time

from .v1_position import WIN_SCORE, Position, other


INFINITY = 10**9


class SearchTimeout(Exception):
    pass


class Searcher:
    def __init__(self, depth: int = 6, width: int = 10, time_budget: float = 2.0) -> None:
        self.depth = max(2, depth)
        self.width = max(4, width)
        self.time_budget = max(0.05, time_budget)

        self.deadline = float("inf")
        self.nodes = 0

    def best_move(self, position: Position, player: int) -> tuple[int, int] | None:
        if position.is_full():
            return None
        greedy = position.candidates(player, max(self.width * 2, 8))


        for row, col in greedy:
            if position.wins_with(row, col, player):
                return row, col
        opponent = other(player)
        blocks = [(row, col) for row, col in greedy if position.wins_with(row, col, opponent)]
        if len(blocks) == 1:
            return blocks[0]


        self.deadline = time.monotonic() + self.time_budget
        best = greedy[0]
        best_value = -INFINITY
        for depth in range(2, self.depth + 1, 2):
            try:
                value, move = self._root(position, player, depth)
            except SearchTimeout:
                break
            if move is not None:
                best, best_value = move, value
            if best_value >= WIN_SCORE // 2:
                break
        return best

    def _root(self, position: Position, player: int, depth: int) -> tuple[int, tuple[int, int]]:
        moves = position.candidates(player, self.width)
        best_value = -INFINITY
        best_move = moves[0]
        alpha = -INFINITY
        for row, col in moves:
            undo = position.play(row, col, player)
            try:
                if undo.win:
                    value = WIN_SCORE
                else:
                    value = -self._negamax(position, depth - 1, -INFINITY, -alpha, other(player))
            finally:
                position.undo(undo)
            if value > best_value:
                best_value, best_move = value, (row, col)
            if best_value > alpha:
                alpha = best_value
        return best_value, best_move

    def _negamax(
        self,
        position: Position,
        depth: int,
        alpha: int,
        beta: int,
        player: int,
    ) -> int:
        self.nodes += 1
        if time.monotonic() > self.deadline:
            raise SearchTimeout
        if depth <= 0:
            return position.evaluate(player)
        moves = position.candidates(player, self.width)
        if not moves:
            return position.evaluate(player)
        best = -INFINITY
        for row, col in moves:
            undo = position.play(row, col, player)
            try:
                if undo.win:
                    value = WIN_SCORE - (self.depth - depth)
                else:
                    value = -self._negamax(position, depth - 1, -beta, -alpha, other(player))
            finally:
                position.undo(undo)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
