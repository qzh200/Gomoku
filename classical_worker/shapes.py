from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .patterns import FIVE_WINDOW, OPEN_THREE_EXTENSION
from .position import DIRECTIONS, EMPTY, Position, in_bounds


Cell = tuple[int, int]
Line = tuple[Cell, ...]

OPEN_FOUR = "open_four"
CLOSED_FOUR = "closed_four"
OPEN_THREE = "open_three"

_OWN = 1
_EMPTY = 0
_OTHER = 2


@dataclass(frozen=True, slots=True)
class Threat:
    kind: str
    stones: frozenset[Cell]
    points: tuple[Cell, ...]
    blocks: tuple[Cell, ...]
    extensions: tuple[tuple[Cell, tuple[Cell, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class ThreatProfile:
    player: int
    threats: tuple[Threat, ...]

    def of(self, kind: str) -> tuple[Threat, ...]:
        return tuple(threat for threat in self.threats if threat.kind == kind)

    @property
    def open_fours(self) -> tuple[Threat, ...]:
        return self.of(OPEN_FOUR)

    @property
    def closed_fours(self) -> tuple[Threat, ...]:
        return self.of(CLOSED_FOUR)

    @property
    def open_threes(self) -> tuple[Threat, ...]:
        return self.of(OPEN_THREE)

    @property
    def fours(self) -> tuple[Threat, ...]:
        return self.open_fours + self.closed_fours

    @property
    def five_points(self) -> frozenset[Cell]:
        points: set[Cell] = set()
        for threat in self.fours:
            points.update(threat.points)
        return frozenset(points)

    @property
    def block_points(self) -> frozenset[Cell]:
        points: set[Cell] = set()
        for threat in self.threats:
            points.update(threat.blocks)
        return frozenset(points)


def threat_profile(position: Position, player: int) -> ThreatProfile:
    threats: list[Threat] = []
    for cells, values in _lines(position, player):
        threats.extend(_fours(cells, values))
        threats.extend(_threes(cells, values))
    return ThreatProfile(player, tuple(threats))


def board_threats(position: Position) -> tuple[ThreatProfile, ThreatProfile]:
    return threat_profile(position, 1), threat_profile(position, 2)


def four_groups(values: Sequence[int]) -> dict[frozenset[int], set[int]]:
    groups: dict[frozenset[int], set[int]] = {}
    for start in range(len(values) - FIVE_WINDOW + 1):
        window = values[start : start + FIVE_WINDOW]
        own = [index for index, value in enumerate(window) if value == _OWN]
        free = [index for index, value in enumerate(window) if value == _EMPTY]
        if len(own) != 4 or len(free) != 1:
            continue
        stones = frozenset(start + index for index in own)
        groups.setdefault(stones, set()).add(start + free[0])
    return groups


def three_groups(values: Sequence[int]) -> dict[frozenset[int], dict[int, list[int]]]:
    text = "".join(str(value) for value in values)
    groups: dict[frozenset[int], dict[int, list[int]]] = {}
    for start in range(len(text) - 5):
        window = text[start : start + 6]
        extension = OPEN_THREE_EXTENSION.get(window)
        if extension is None:
            continue
        stones = frozenset(start + index for index in range(6) if window[index] == "1")
        ends = [
            start + index
            for index in range(6)
            if window[index] == "0" and index != extension
        ]
        group = groups.setdefault(stones, {})
        group.setdefault(start + extension, ends)
    return groups


def _lines(position: Position, player: int) -> list[tuple[Line, list[int]]]:
    grid = position.grid
    seen: set[tuple[int, int, int, int]] = set()
    found: list[tuple[Line, list[int]]] = []
    for row, col, stone in position.stones:
        if stone != player:
            continue
        for dr, dc in DIRECTIONS:
            start_row, start_col = row, col
            while in_bounds(start_row - dr, start_col - dc):
                start_row -= dr
                start_col -= dc
            key = (dr, dc, start_row, start_col)
            if key in seen:
                continue
            seen.add(key)
            cells: list[Cell] = []
            values: list[int] = []
            current_row, current_col = start_row, start_col
            while in_bounds(current_row, current_col):
                value = grid[current_row][current_col]
                cells.append((current_row, current_col))
                if value == player:
                    values.append(_OWN)
                elif value == EMPTY:
                    values.append(_EMPTY)
                else:
                    values.append(_OTHER)
                current_row += dr
                current_col += dc
            found.append((tuple(cells), values))
    return found


def _fours(line: Line, values: list[int]) -> list[Threat]:
    threats: list[Threat] = []
    for stones, points in four_groups(values).items():
        cells = tuple(sorted(line[index] for index in points))
        kind = OPEN_FOUR if len(cells) >= 2 else CLOSED_FOUR
        threats.append(
            Threat(
                kind,
                frozenset(line[index] for index in stones),
                cells,
                () if kind == OPEN_FOUR else cells,
            )
        )
    return threats


def _threes(line: Line, values: list[int]) -> list[Threat]:
    threats: list[Threat] = []
    for stones, extensions in three_groups(values).items():
        points = tuple(sorted(line[index] for index in extensions))
        blocks = set.intersection(
            *[{extension, *ends} for extension, ends in extensions.items()]
        )
        threats.append(
            Threat(
                OPEN_THREE,
                frozenset(line[index] for index in stones),
                points,
                tuple(sorted(line[index] for index in blocks)),
                tuple(
                    (line[extension], tuple(line[index] for index in ends))
                    for extension, ends in sorted(extensions.items())
                ),
            )
        )
    return threats
