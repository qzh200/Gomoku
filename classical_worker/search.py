"""The search: negamax with alpha-beta pruning over the greedy moves.

Negamax asks every node the same question ("how good is this for the side to
move?") and flips the sign of the answer on the way back, so one piece of code
covers both players. Alpha-beta cuts the branches that cannot change the
answer, and the greedy ordering of the moves puts the loud ones first, which
is what makes the cuts happen early.

The depth grows from two until the time budget runs out, so the move of the
last finished depth is always playable.
"""

from __future__ import annotations

import time

from .position import WIN_SCORE, Position, other

# A value no real score reaches.
INFINITY = 10**9


class SearchTimeout(Exception):
    """The time budget of the search ran out."""


class Searcher:
    """Picks a move with negamax and alpha-beta pruning."""

    def __init__(self, depth: int = 6, width: int = 10, time_budget: float = 2.0) -> None:
        self.depth = max(2, depth)
        self.width = max(4, width)
        self.time_budget = max(0.05, time_budget)
        # best_move sets the deadline; a direct call to the search has none.
        self.deadline = float("inf")
        self.nodes = 0

    def best_move(self, position: Position, player: int) -> tuple[int, int] | None:
        """Return the move to play, or None when the board is full."""
        if position.is_full():
            return None
        greedy = position.candidates(player, max(self.width * 2, 8))

        # Greedy first: win at once, and stop a five that is one move away.
        for row, col in greedy:
            if position.wins_with(row, col, player):
                return row, col
        opponent = other(player)
        blocks = [(row, col) for row, col in greedy if position.wins_with(row, col, opponent)]
        if len(blocks) == 1:
            return blocks[0]

        # Negamax, deeper every round.
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
        """Search every move of the root and return the best one."""
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
                # A timed out search must leave the board as it found it.
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
        """Return the value of the position for the side to move."""
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
                    # The shallower the win, the higher its value, so the
                    # engine finishes the game instead of wandering.
                    value = WIN_SCORE - (self.depth - depth)
                else:
                    value = -self._negamax(position, depth - 1, -beta, -alpha, other(player))
            finally:
                # A timed out search must leave the board as it found it.
                position.undo(undo)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
