"""The search: negamax with alpha-beta pruning over the greedy moves.


"""

from __future__ import annotations

import time




class SearchTimeout(Exception):
    """The time budget of the search ran out."""


class Searcher:

        self.depth = max(2, depth)
        self.width = max(4, width)
        self.time_budget = max(0.05, time_budget)
        self.nodes = 0

        """Return the move to play, or None when the board is full."""
        if position.is_full():
            return None

        opponent = other(player)

            try:
            except SearchTimeout:
                break
            if move is not None:
                break
        return best

    def _root(self, position: Position, player: int, depth: int) -> tuple[int, tuple[int, int]]:
        best_value = -INFINITY
        best_move = moves[0]
        alpha = -INFINITY
            try:
                if undo.win:
                else:
            finally:
                # A timed out search must leave the board as it found it.
                position.undo(undo)
            if value > best_value:
        return best_value, best_move

        self,
        position: Position,
        depth: int,
        alpha: int,
        beta: int,
        player: int,
    ) -> int:
        """Return the value of the position for the side to move."""
        if depth <= 0:
            return position.evaluate(player)
        if not moves:
        best = -INFINITY
            try:
                if undo.win:
                else:
            finally:
                # A timed out search must leave the board as it found it.
                position.undo(undo)
            if value > best:
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
