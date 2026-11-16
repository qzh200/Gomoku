from __future__ import annotations

from dataclasses import dataclass


MATE_SCORE = 10**9
MATE_THRESHOLD = MATE_SCORE - 1000

EXACT = 0
LOWER = 1
UPPER = 2


SIDE_KEY = {1: 0x9E3779B97F4A7C15, 2: 0xC2B2AE3D27D4EB4F}

Move = tuple[int, int]


@dataclass(slots=True)
class Entry:
    depth: int
    value: int
    flag: int
    move: Move | None
    age: int


class TranspositionTable:
    def __init__(self, max_entries: int = 200_000) -> None:
        self.max_entries = max(1024, max_entries)
        self.table: dict[int, Entry] = {}
        self.age = 0
        self.probes = 0
        self.hits = 0

    def new_generation(self) -> None:
        self.age += 1
        if len(self.table) > self.max_entries:
            self.table.clear()

    def clear(self) -> None:
        self.table.clear()
        self.age = 0
        self.probes = 0
        self.hits = 0

    def probe(self, key: int) -> Entry | None:
        self.probes += 1
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(
        self,
        key: int,
        depth: int,
        value: int,
        flag: int,
        move: Move | None,
    ) -> None:
        if len(self.table) >= self.max_entries:
            self.table.clear()
        old = self.table.get(key)
        if old is not None and old.age == self.age and old.depth > depth:
            return
        self.table[key] = Entry(depth, value, flag, move, self.age)


def to_table(value: int, ply: int) -> int:
    if value > MATE_THRESHOLD:
        return value + ply
    if value < -MATE_THRESHOLD:
        return value - ply
    return value


def from_table(value: int, ply: int) -> int:
    if value > MATE_THRESHOLD:
        return value - ply
    if value < -MATE_THRESHOLD:
        return value + ply
    return value
