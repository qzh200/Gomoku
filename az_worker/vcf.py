from __future__ import annotations

import time

from .board import BOARD_SIZE
from .threats import (
    STRIDE,
    board_masks,
    bits,
    five_mask,
    near_mask,
    open_four_mask,
    open_three_mask,
)


VCF = "vcf"

VCT = "vct"

DEFAULT_DEPTH = 10

DEFAULT_NODES = 4_000


class _Abort(Exception):
    pass


def _count(mask: int) -> int:
    return mask.bit_count()


def _single(mask: int) -> int:
    return mask.bit_length() - 1


class Solver:
    def __init__(self, max_depth: int = DEFAULT_DEPTH, max_nodes: int = DEFAULT_NODES) -> None:
        self.max_depth = max(2, max_depth)
        self.max_nodes = max(1, max_nodes)
        self.nodes = 0
        self.deadline = float("inf")
        self.table: dict[tuple[int, int, int], bool] = {}

    def winning_move(
        self,
        board,
        player: int,
        deadline: float,
        mode: str = VCF,
    ) -> tuple[int, int] | None:
        own, other, empty = board_masks(board, player)
        immediate = five_mask(own, empty)
        if immediate:
            index = _single(immediate)
            return divmod(index, STRIDE)

        self.nodes = 0
        self.deadline = deadline
        self.table = {}
        start = 2 if mode == VCF else 1
        try:
            for depth in range(start, self.max_depth + 1):
                move = self._root(own, other, empty, depth, mode)
                if move is not None:
                    return move
        except _Abort:
            return None
        return None

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise _Abort
        if self.nodes % 16 == 0 and time.monotonic() > self.deadline:
            raise _Abort

    def _root(
        self,
        own: int,
        other: int,
        empty: int,
        depth: int,
        mode: str,
    ) -> tuple[int, int] | None:
        for index in self._attack_moves(own, other, empty, mode):
            self._tick()
            bit = 1 << index
            if five_mask(own, empty) & bit:
                return divmod(index, STRIDE)
            if self._defend(own | bit, other, empty & ~bit, depth - 1, mode):
                return divmod(index, STRIDE)
        return None

    def _attack(self, own: int, other: int, empty: int, depth: int, mode: str) -> bool:
        if five_mask(own, empty):
            return True
        if five_mask(other, empty):
            return False
        if depth <= 0:
            return False
        key = (own, other, depth)
        known = self.table.get(key)
        if known is not None:
            return known
        for index in self._attack_moves(own, other, empty, mode):
            self._tick()
            bit = 1 << index
            if five_mask(own, empty) & bit:
                self.table[key] = True
                return True
            if self._defend(own | bit, other, empty & ~bit, depth - 1, mode):
                self.table[key] = True
                return True
        self.table[key] = False
        return False

    def _defend(self, own: int, other: int, empty: int, depth: int, mode: str) -> bool:
        defender_five = five_mask(other, empty)
        if defender_five:
            return False
        attacker_five = five_mask(own, empty)
        if _count(attacker_five) >= 2:
            return True
        if _count(attacker_five) == 1:
            return self._reply(own, other, empty, attacker_five, depth, mode)
        if mode != VCT:
            return False
        fours = open_four_mask(own, empty)
        if not fours:
            return False
        if _count(fours) == 1:
            return self._reply(own, other, empty, fours, depth, mode)
        for reply in bits(fours):
            self._tick()
            bit = 1 << reply
            if five_mask(other, empty) & bit:
                return False
            if not self._attack(own, other | bit, empty & ~bit, depth - 1, mode):
                return False
        return True

    def _reply(
        self,
        own: int,
        other: int,
        empty: int,
        cells: int,
        depth: int,
        mode: str,
    ) -> bool:
        bit = 1 << _single(cells)
        if five_mask(other, empty) & bit:
            return False
        return self._attack(own, other | bit, empty & ~bit, depth - 1, mode)

    def _attack_moves(self, own: int, other: int, empty: int, mode: str) -> list[int]:
        near = near_mask(own | other, empty)
        attacker_five = five_mask(own, empty)
        scored: list[tuple[int, int]] = []
        for index in bits(near):
            bit = 1 << index
            if attacker_five & bit:
                scored.append((1 << 30, index))
                continue
            placed_own = own | bit
            placed_empty = empty & ~bit
            threats = _count(five_mask(placed_own, placed_empty))
            if threats >= 2:
                scored.append((1 << 28, index))
            elif threats == 1:
                score = 1 << 20
                if mode == VCT:
                    score += _count(open_four_mask(placed_own, placed_empty))
                scored.append((score, index))
            elif mode == VCT:
                fours = _count(open_four_mask(placed_own, placed_empty))
                if fours:
                    scored.append(((1 << 12) + fours, index))
        scored.sort(reverse=True)
        return [index for _, index in scored]


def winning_move(
    board,
    player: int,
    budget: float,
    mode: str = VCF,
    max_depth: int = DEFAULT_DEPTH,
    max_nodes: int = DEFAULT_NODES,
) -> tuple[int, int] | None:
    if budget <= 0:
        return None
    solver = Solver(max_depth=max_depth, max_nodes=max_nodes)
    deadline = time.monotonic() + budget
    move = solver.winning_move(board, player, deadline, mode)
    if move is None:
        return None
    row, col = move
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return None
    return row, col


class PlaySolver:
    def __init__(
        self,
        vcf_budget: float = 0.05,
        vct_budget: float = 0.05,
        max_depth: int = DEFAULT_DEPTH,
        max_nodes: int = DEFAULT_NODES,
    ) -> None:
        self.vcf_budget = max(0.0, vcf_budget)
        self.vct_budget = max(0.0, vct_budget)
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def move(self, board, player: int) -> tuple[int, int] | None:
        own, _, empty = board_masks(board, player)
        immediate = five_mask(own, empty)
        if immediate:
            return divmod(immediate.bit_length() - 1, STRIDE)
        if self.vcf_budget > 0:
            move = winning_move(
                board, player, self.vcf_budget, VCF, self.max_depth, self.max_nodes
            )
            if move is not None:
                return move
        if self.vct_budget > 0:
            if not (open_four_mask(own, empty) or open_three_mask(own, empty)):
                return None
            return winning_move(
                board, player, self.vct_budget, VCT, self.max_depth, self.max_nodes
            )
        return None
