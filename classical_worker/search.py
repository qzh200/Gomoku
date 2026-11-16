"""The search: iterative deepening with a principal variation search.

Every node asks the same question ("how good is this for the side to move?")
and the answer changes sign on the way back. The first move is searched with a
full window; the rest get a zero window and are re-searched only when they
beat alpha. The transposition table keeps the value of a position reached by
another order, and the threat ordering puts the forced moves first, which is
what makes the windows close early.

VCF and VCT run before the main search: a forced win found there is played at
once, and the main search keeps the fallback when they find nothing.
"""

from __future__ import annotations

import time

from .position import BOARD_SIZE, Position, other
from .threats import VCF, VCT, forced_moves
from .tt import (
    EXACT,
    LOWER,
    MATE_SCORE,
    MATE_THRESHOLD,
    SIDE_KEY,
    UPPER,
    TranspositionTable,
    from_table,
    to_table,
)

# A value above every mate score, used for the open window.
INFINITY = 10**10

# The share of the budget the forced searches may use.
DEFAULT_VCF_BUDGET = 0.10
DEFAULT_VCT_BUDGET = 0.0


class SearchTimeout(Exception):
    """The time budget of the search ran out."""


class Searcher:
    """Picks a move with iterative deepening and a principal variation search."""

    def __init__(
        self,
        depth: int = 8,
        width: int = 12,
        time_budget: float = 2.0,
        vcf_budget: float = DEFAULT_VCF_BUDGET,
        vct_budget: float = DEFAULT_VCT_BUDGET,
        tt_entries: int = 200_000,
        max_nodes: int | None = None,
    ) -> None:
        self.depth = max(2, depth)
        self.width = max(4, width)
        self._limit = self.width
        self.time_budget = max(0.05, time_budget)
        self.vcf_budget = max(0.0, vcf_budget)
        self.vct_budget = max(0.0, vct_budget)
        self.max_nodes = max_nodes
        # The table, the history and the killers live across turns. The keys
        # are complete positions, so a hit is never stale.
        self.table = TranspositionTable(tt_entries)
        self.history: dict[tuple[int, int, int], int] = {}
        self.killers: dict[int, list[tuple[int, int]]] = {}
        self.nodes = 0
        self.depth_reached = 0
        self.deadline = float("inf")
        self.soft_deadline = float("inf")

    def best_move(
        self,
        position: Position,
        player: int,
        depth: int | None = None,
        width: int | None = None,
        time_budget: float | None = None,
    ) -> tuple[int, int] | None:
        """Return the move to play, or None when the board is full."""
        if position.is_full():
            return None
        if not position.stones:
            return (BOARD_SIZE // 2, BOARD_SIZE // 2)

        max_depth = max(2, int(self.depth if depth is None else depth))
        limit = max(4, int(self.width if width is None else width))
        self._limit = limit
        budget = max(0.05, float(self.time_budget if time_budget is None else time_budget))
        deadline = time.monotonic() + budget * 0.95
        self.table.new_generation()
        self.killers = {}
        self.history = {}
        self.nodes = 0
        self.depth_reached = 0

        neighbours = position.neighbours()
        if not neighbours:
            return None

        # A five now, and a five of the opponent, are decided without search.
        own_five = [cell for cell in neighbours if position.wins_with(*cell, player)]
        if own_five:
            return own_five[0]
        opponent = other(player)
        opponent_five = [cell for cell in neighbours if position.wins_with(*cell, opponent)]

        if len(opponent_five) == 1:
            return opponent_five[0]

        # Two fives cannot both be blocked. Only a win of our own saves the
        # position, so look for one before returning a lost move.
        if len(opponent_five) >= 2:
            move = VCF(max_depth=max(4, max_depth)).best_move(
                position, player, min(deadline, time.monotonic() + budget * self.vcf_budget)
            )
            return move or opponent_five[0]

        if self.vcf_budget > 0:
            vcf_deadline = min(deadline, time.monotonic() + budget * self.vcf_budget)
            move = VCF(max_depth=max(4, max_depth)).best_move(position, player, vcf_deadline)
            if move is not None:
                return move

        if self.vct_budget > 0:
            vct_deadline = min(deadline, time.monotonic() + budget * self.vct_budget)
            move = VCT(max_depth=max(2, max_depth // 2)).best_move(position, player, vct_deadline)
            if move is not None:
                return move

        self.deadline = deadline
        self.soft_deadline = time.monotonic() + budget * 0.85
        moves = forced_moves(position, player, limit * 2, None, (), self.history, 0)
        if not moves:
            return None
        best = moves[0]
        for current in range(2, max_depth + 1):
            try:
                value, move = self._root(position, player, current)
            except SearchTimeout:
                break
            if move is not None:
                best = move
                self.depth_reached = current
            if value >= MATE_THRESHOLD:
                break
            if time.monotonic() > self.soft_deadline:
                break
        return best

    # -- the tree ---------------------------------------------------------

    def _root(self, position: Position, player: int, depth: int) -> tuple[int, tuple[int, int]]:
        """Search every root move and return the best value and move."""
        moves = forced_moves(
            position,
            player,
            self._limit * 2,
            None,
            tuple(self.killers.get(0, ())),
            self.history,
            0,
        )
        best_value = -INFINITY
        best_move = moves[0]
        alpha = -INFINITY
        for index, move in enumerate(moves):
            undo = position.play(*move, player)
            try:
                if undo.win:
                    value = MATE_SCORE - 1
                elif index == 0:
                    value = -self._pvs(position, depth - 1, -INFINITY, -alpha, other(player), 1)
                else:
                    value = -self._pvs(position, depth - 1, -alpha - 1, -alpha, other(player), 1)
                    if value > alpha:
                        value = -self._pvs(position, depth - 1, -INFINITY, -alpha, other(player), 1)
            finally:
                # A timed out search must leave the board as it found it.
                position.undo(undo)
            if value > best_value:
                best_value, best_move = value, move
            if value > alpha:
                alpha = value
        key = position.hash ^ SIDE_KEY[player]
        self.table.store(key, depth, to_table(best_value, 0), EXACT, best_move)
        return best_value, best_move

    def _pvs(
        self,
        position: Position,
        depth: int,
        alpha: int,
        beta: int,
        player: int,
        ply: int,
    ) -> int:
        """Return the value of the position for the side to move."""
        self._tick()
        if depth <= 0:
            return position.evaluate(player)
        if position.is_full():
            return 0

        key = position.hash ^ SIDE_KEY[player]
        entry = self.table.probe(key)
        tt_move = None
        if entry is not None:
            tt_move = entry.move
            if entry.depth >= depth:
                value = from_table(entry.value, ply)
                if entry.flag == EXACT:
                    return value
                if entry.flag == LOWER:
                    alpha = max(alpha, value)
                elif entry.flag == UPPER:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value

        moves = forced_moves(
            position,
            player,
            self._limit,
            tt_move,
            tuple(self.killers.get(ply, ())),
            self.history,
            ply,
        )
        if not moves:
            return 0

        alpha_original = alpha
        best = -INFINITY
        best_move = None
        for index, move in enumerate(moves):
            undo = position.play(*move, player)
            try:
                if undo.win:
                    value = MATE_SCORE - (ply + 1)
                elif index == 0:
                    value = -self._pvs(position, depth - 1, -beta, -alpha, other(player), ply + 1)
                else:
                    value = -self._pvs(position, depth - 1, -alpha - 1, -alpha, other(player), ply + 1)
                    if alpha < value < beta:
                        value = -self._pvs(position, depth - 1, -beta, -alpha, other(player), ply + 1)
            finally:
                # A timed out search must leave the board as it found it.
                position.undo(undo)
            if value > best:
                best, best_move = value, move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if not undo.win:
                    killers = self.killers.setdefault(ply, [])
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    self.history[(player, move[0], move[1])] = (
                        self.history.get((player, move[0], move[1]), 0) + depth * depth
                    )
                break

        if best <= alpha_original:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.table.store(key, depth, to_table(best, ply), flag, best_move)
        return best

    def _tick(self) -> None:
        """Count a node and stop when the budget is spent."""
        self.nodes += 1
        if self.max_nodes is not None and self.nodes >= self.max_nodes:
            raise SearchTimeout
        if self.nodes % 16 == 0 and time.monotonic() > self.deadline:
            raise SearchTimeout
