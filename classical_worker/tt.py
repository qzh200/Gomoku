"""The transposition table.

The search reaches the same board by different orders of moves. The table
maps a Zobrist hash to what an earlier, at least as deep, search concluded
about it. Values that say "mate in N" are stored as seen from the node, not
from the root, so a transposition found at another depth is still valid.
"""

from __future__ import annotations

from dataclasses import dataclass

# A value above every static score, and the margin that identifies a mate.
MATE_SCORE = 10**9
MATE_THRESHOLD = MATE_SCORE - 1000

EXACT = 0
LOWER = 1
UPPER = 2

# The hash of the side to move, so the same board for the two players is two
# different search nodes.
SIDE_KEY = {1: 0x9E3779B97F4A7C15, 2: 0xC2B2AE3D27D4EB4F}

Move = tuple[int, int]


@dataclass(slots=True)
class Entry:
    """What the search knows about one position."""

    depth: int
    value: int
    flag: int
    move: Move | None
    age: int


class TranspositionTable:
    """A bounded map from a position hash to an ``Entry``."""

    def __init__(self, max_entries: int = 200_000) -> None:
        self.max_entries = max(1024, max_entries)
        self.table: dict[int, Entry] = {}
        self.age = 0
        self.probes = 0
        self.hits = 0

    def new_generation(self) -> None:
        """Start a new decision; old entries become replaceable."""
        self.age += 1
        if len(self.table) > self.max_entries:
            self.table.clear()

    def clear(self) -> None:
        self.table.clear()
        self.age = 0
        self.probes = 0
        self.hits = 0

    def probe(self, key: int) -> Entry | None:
        """Return the entry of a position, if any."""
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
        """Save a search result, keeping deeper entries over shallower ones."""
        if len(self.table) >= self.max_entries:
            self.table.clear()
        old = self.table.get(key)
        if old is not None and old.age == self.age and old.depth > depth:
            return
        self.table[key] = Entry(depth, value, flag, move, self.age)


def to_table(value: int, ply: int) -> int:
    """Return a mate score in the distance-from-this-node form."""
    if value > MATE_THRESHOLD:
        return value + ply
    if value < -MATE_THRESHOLD:
        return value - ply
    return value


def from_table(value: int, ply: int) -> int:
    """Return a stored mate score as a score from the root."""
    if value > MATE_THRESHOLD:
        return value - ply
    if value < -MATE_THRESHOLD:
        return value + ply
    return value
