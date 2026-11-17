from __future__ import annotations

import time
from dataclasses import dataclass

from .patterns import OPEN_THREE_PATTERNS
from .position import DIRECTIONS, EMPTY, Position, in_bounds, other
from .tt import SIDE_KEY


THREAT_RADIUS = 4
POW3_8 = tuple(3**index for index in range(8))


OWN = "1"
FREE = "0"


@dataclass(frozen=True, slots=True)
class Threat:
    five: bool
    five_points: int
    open_threes: int

    @property
    def rank(self) -> int:
        if self.five:
            return 1_000_000_000
        if self.five_points >= 2:
            return 100_000_000
        if self.five_points >= 1 and self.open_threes >= 1:
            return 90_000_000
        if self.open_threes >= 2:
            return 80_000_000
        if self.five_points == 1:
            return 70_000_000
        if self.open_threes == 1:
            return 60_000_000
        return 0


def _classify_line(text: str) -> tuple[int, int, int]:
    five = 0
    five_points: set[int] = set()
    open_threes = 0


    for start in range(5):
        if not start <= 4 <= start + 4:
            continue
        window = text[start : start + 5]
        ones = window.count(OWN)
        if ones == 5:
            five = 1
        elif ones == 4 and window.count(FREE) == 1:
            five_points.add(start + window.index(FREE))


    for start in range(4):
        window = text[start : start + 6]
        if window in OPEN_THREE_PATTERNS and window[4 - start] == OWN:
            open_threes = 1
            break

    return five, len(five_points), open_threes


def _build_threat_table() -> tuple[tuple[int, int, int], ...]:
    table = []
    for code in range(3**8):
        cells = [FREE] * 9
        cells[4] = OWN
        shift = 0
        for index in range(9):
            if index == 4:
                continue
            digit = (code // POW3_8[shift]) % 3
            cells[index] = "012"[digit]
            shift += 1
        table.append(_classify_line("".join(cells)))
    return tuple(table)


THREAT_TABLE = _build_threat_table()


def _line_code(position: Position, row: int, col: int, player: int, dr: int, dc: int) -> int:
    code = 0
    shift = 0
    grid = position.grid
    for step in range(-THREAT_RADIUS, THREAT_RADIUS + 1):
        if step == 0:
            continue
        r, c = row + step * dr, col + step * dc
        if not in_bounds(r, c):
            value = 2
        else:
            cell = grid[r][c]
            if cell == EMPTY:
                value = 0
            elif cell == player:
                value = 1
            else:
                value = 2
        code += value * POW3_8[shift]
        shift += 1
    return code


def classify(position: Position, row: int, col: int, player: int) -> Threat:
    five = 0
    five_points = 0
    open_threes = 0
    for dr, dc in DIRECTIONS:
        line_five, line_points, line_threes = THREAT_TABLE[
            _line_code(position, row, col, player, dr, dc)
        ]
        five |= line_five
        five_points += line_points
        open_threes += line_threes
    return Threat(bool(five), five_points, open_threes)


def five_moves(position: Position, player: int) -> list[tuple[int, int]]:
    return [cell for cell in position.neighbours() if position.wins_with(*cell, player)]


def _scored(
    position: Position,
    player: int,
    cells: list[tuple[int, int]],
    tt_move: tuple[int, int] | None,
    killers: tuple[tuple[int, int], ...],
    history: dict[tuple[int, int, int], int] | None,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    opponent = other(player)
    scored = []
    wins = []
    blocks = []
    for row, col in cells:
        attack = classify(position, row, col, player)
        defence = classify(position, row, col, opponent)
        if attack.five:
            wins.append((row, col))
        if defence.five:
            blocks.append((row, col))
        attack_rank = attack.rank
        defence_rank = defence.rank
        history_score = 0 if history is None else history.get((player, row, col), 0)
        killer = 1 if (row, col) in killers else 0
        scored.append(
            (
                (
                    1 if (row, col) == tt_move else 0,
                    max(attack_rank, defence_rank),
                    attack_rank + defence_rank,
                    killer,
                    history_score,
                    -row,
                    -col,
                ),
                (row, col),
            )
        )
    return scored, wins, blocks


def forced_moves(
    position: Position,
    player: int,
    limit: int,
    tt_move: tuple[int, int] | None = None,
    killers: tuple[tuple[int, int], ...] = (),
    history: dict[tuple[int, int, int], int] | None = None,
) -> list[tuple[int, int]]:
    neighbours = position.neighbours()
    if not neighbours:
        return []
    scored, wins, blocks = _scored(position, player, neighbours, tt_move, killers, history)
    if wins:
        return wins[:limit]
    if blocks:
        return blocks[:limit]
    scored.sort(reverse=True)
    return [cell for _, cell in scored[:limit]]


class _Abort(Exception):
    pass


class _ForcedSearch:
    def __init__(self, max_depth: int, max_nodes: int) -> None:
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes = 0
        self.deadline = float("inf")
        self.tt: dict[tuple[int, int], bool] = {}

    def _start(self, deadline: float) -> None:
        self.nodes = 0
        self.deadline = deadline
        self.tt = {}

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise _Abort
        if self.nodes % 8 == 0 and time.monotonic() > self.deadline:
            raise _Abort

    def _key(self, position: Position, player: int, depth: int) -> tuple[int, int]:
        return (position.hash ^ SIDE_KEY[player], depth)


class VCF(_ForcedSearch):
    def __init__(self, max_depth: int = 10, max_nodes: int = 50_000) -> None:
        super().__init__(max_depth, max_nodes)

    def best_move(self, position: Position, attacker: int, deadline: float) -> tuple[int, int] | None:
        self._start(deadline)
        try:
            for depth in range(2, self.max_depth + 1, 2):
                move = self._root(position, attacker, depth)
                if move is not None:
                    return move
        except _Abort:
            return None
        return None

    def _root(self, position: Position, attacker: int, depth: int) -> tuple[int, int] | None:
        for move in self._attack_moves(position, attacker):
            self._tick()
            undo = position.play(*move, attacker)
            try:
                if undo.win:
                    return move
                if self._wins_after(position, attacker, depth - 1):
                    return move
            finally:
                position.undo(undo)
        return None

    def _wins_after(self, position: Position, attacker: int, depth: int) -> bool:
        opponent = other(attacker)
        if five_moves(position, opponent):
            return False
        winning = five_moves(position, attacker)
        if len(winning) >= 2:
            return True
        if len(winning) != 1:
            return False
        block = winning[0]
        undo = position.play(*block, opponent)
        try:
            if undo.win:
                return False
            return self._search(position, attacker, depth, 2)
        finally:
            position.undo(undo)

    def _search(self, position: Position, attacker: int, depth: int, ply: int) -> bool:
        self._tick()
        if depth <= 0:
            return False
        key = self._key(position, attacker, depth)
        known = self.tt.get(key)
        if known is not None:
            return known
        for move in self._attack_moves(position, attacker):
            self._tick()
            undo = position.play(*move, attacker)
            try:
                if undo.win:
                    self.tt[key] = True
                    return True
                if self._wins_after(position, attacker, depth - 1):
                    self.tt[key] = True
                    return True
            finally:
                position.undo(undo)
        self.tt[key] = False
        return False

    @staticmethod
    def _attack_moves(position: Position, attacker: int) -> list[tuple[int, int]]:
        scored = []
        for cell in position.neighbours():
            if position.wins_with(*cell, attacker):
                scored.append((10**6, cell))
                continue
            threat = classify(position, *cell, attacker)
            if threat.five_points:
                scored.append((threat.five_points, cell))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cell for _, cell in scored]
