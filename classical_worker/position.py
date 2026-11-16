"""The position of a Gomoku game and its score.

The board keeps the window scores of both players and a Zobrist hash up to
date while stones are played and taken back. A stone only changes the four
lines that cross it, so a move updates at most 24 windows and evaluation is a
dictionary lookup.

Windows are scored by :mod:`classical_worker.patterns`; the board itself knows
nothing about shapes.
"""

from __future__ import annotations

import random

from .patterns import POW3, WINDOW, WINDOW_SCORES

BOARD_SIZE = 15
COLUMN_LABELS = "ABCDEFGHIJKLMNO"

# The cells of the board.
EMPTY = 0
BLACK = 1
WHITE = 2

# The four directions of a line: across, down, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# The score of a five, above every other window.
WIN_SCORE = 10_000_000

# The Zobrist values of a stone, fixed so two runs see the same hashes.
_RANDOM = random.Random(0xC0FFEE)
ZOBRIST = tuple(
    tuple(_RANDOM.getrandbits(64) for _ in range(BOARD_SIZE * BOARD_SIZE))
    for _ in range(3)
)


def other(player: int) -> int:
    """Return the opponent of a player."""
    return BLACK if player == WHITE else WHITE


def in_bounds(row: int, col: int) -> bool:
    """Report whether a cell is on the board."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def coordinate_of(row: int, col: int) -> str:
    """Return the label of a cell, like "M5", the way the board shows it."""
    if not in_bounds(row, col):
        return "?"
    return f"{COLUMN_LABELS[col]}{row + 1}"


class Undo:
    """What playing one stone changed, so the move can be taken back."""

    __slots__ = ("row", "col", "player", "win", "deltas")

    def __init__(
        self,
        row: int,
        col: int,
        player: int,
        win: bool,
        deltas: tuple[tuple[int, int], ...],
    ) -> None:
        self.row = row
        self.col = col
        self.player = player
        self.win = win
        # The window codes before and after the move, in enumeration order.
        self.deltas = deltas


class Position:
    """A board with the window scores of both players and a Zobrist hash."""

    def __init__(self, rows: list[list[int]] | None = None) -> None:
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        if rows is not None:
            for row, values in enumerate(rows[:BOARD_SIZE]):
                for col, value in enumerate(values[:BOARD_SIZE]):
                    self.grid[row][col] = int(value)
        # The stones in the order they were played, so the neighbours of a
        # stone are found without scanning the board.
        self.stones: list[tuple[int, int, int]] = []
        self.totals = {BLACK: 0, WHITE: 0}
        self.hash = 0
        self._neighbour_cache: list[tuple[int, int]] | None = None
        self._recount()

    def copy(self) -> "Position":
        """Return a copy that shares nothing with this position."""
        return Position(self.grid)

    # -- the windows of the board ----------------------------------------

    @staticmethod
    def _all_windows():
        """Yield every window that lies entirely on the board."""
        for dr, dc in DIRECTIONS:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    end_row = row + (WINDOW - 1) * dr
                    end_col = col + (WINDOW - 1) * dc
                    if in_bounds(end_row, end_col):
                        yield row, col, dr, dc

    def _code_at(self, row: int, col: int, dr: int, dc: int) -> int:
        """Return the code of the window that starts at a cell."""
        grid = self.grid
        code = 0
        for step in range(WINDOW):
            code += grid[row + step * dr][col + step * dc] * POW3[step]
        return code

    def _windows_through(self, row: int, col: int):
        """Yield the windows that contain a cell."""
        for dr, dc in DIRECTIONS:
            for offset in range(WINDOW):
                start_row = row - offset * dr
                start_col = col - offset * dc
                end_row = start_row + (WINDOW - 1) * dr
                end_col = start_col + (WINDOW - 1) * dc
                if in_bounds(start_row, start_col) and in_bounds(end_row, end_col):
                    yield start_row, start_col, dr, dc

    def _codes_through(self, row: int, col: int) -> list[int]:
        """Return the codes of the windows that contain a cell."""
        return [self._code_at(*window) for window in self._windows_through(row, col)]

    def _recount(self) -> None:
        """Rebuild the scores, the stones and the hash from the grid."""
        self.stones = []
        self.totals = {BLACK: 0, WHITE: 0}
        self.hash = 0
        self._neighbour_cache = None
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                player = self.grid[row][col]
                if player != EMPTY:
                    self.stones.append((row, col, player))
                    self.hash ^= ZOBRIST[player][row * BOARD_SIZE + col]
        for row, col, dr, dc in self._all_windows():
            black, white = WINDOW_SCORES[self._code_at(row, col, dr, dc)]
            self.totals[BLACK] += black
            self.totals[WHITE] += white

    def recount(self) -> None:
        """Rebuild the derived state; the tests use it to check the deltas."""
        self._recount()

    # -- the score of the board ------------------------------------------

    def evaluate(self, player: int) -> int:
        """Return the score of the board for one player."""
        return self.totals[player] - self.totals[other(player)]

    # -- playing and taking back ------------------------------------------

    def wins_with(self, row: int, col: int, player: int) -> bool:
        """Report whether a stone on this cell makes five in a row."""
        for dr, dc in DIRECTIONS:
            length = 1 + self._span(row, col, dr, dc, player)[0] + self._span(row, col, -dr, -dc, player)[0]
            if length >= 5:
                return True
        return False

    def _span(self, row: int, col: int, dr: int, dc: int, player: int) -> tuple[int, int, int]:
        """Return the run of a player next to a cell, and the cell past it."""
        length = 0
        r, c = row + dr, col + dc
        while in_bounds(r, c) and self.grid[r][c] == player:
            length += 1
            r, c = r + dr, c + dc
        return length, r, c

    def play(self, row: int, col: int, player: int) -> Undo:
        """Play a stone and return what taking it back needs."""
        if self.grid[row][col] != EMPTY:
            raise ValueError(f"格 {coordinate_of(row, col)} 已有棋子")
        win = self.wins_with(row, col, player)
        before = self._codes_through(row, col)
        self.grid[row][col] = player
        after = self._codes_through(row, col)
        for old, new in zip(before, after):
            old_black, old_white = WINDOW_SCORES[old]
            new_black, new_white = WINDOW_SCORES[new]
            self.totals[BLACK] += new_black - old_black
            self.totals[WHITE] += new_white - old_white
        self.hash ^= ZOBRIST[player][row * BOARD_SIZE + col]
        self.stones.append((row, col, player))
        self._neighbour_cache = None
        return Undo(row, col, player, win, tuple(zip(before, after)))

    def undo(self, undo: Undo) -> None:
        """Take a stone back."""
        self.grid[undo.row][undo.col] = EMPTY
        for old, new in undo.deltas:
            old_black, old_white = WINDOW_SCORES[old]
            new_black, new_white = WINDOW_SCORES[new]
            self.totals[BLACK] += old_black - new_black
            self.totals[WHITE] += old_white - new_white
        self.hash ^= ZOBRIST[undo.player][undo.row * BOARD_SIZE + undo.col]
        self.stones.pop()
        self._neighbour_cache = None

    def is_full(self) -> bool:
        """Report whether no cell is free."""
        return len(self.stones) == BOARD_SIZE * BOARD_SIZE

    # -- the moves worth looking at ---------------------------------------

    def neighbours(self, radius: int = 1) -> list[tuple[int, int]]:
        """Return the empty cells within ``radius`` of a stone, sorted."""
        if radius == 1 and self._neighbour_cache is not None:
            return self._neighbour_cache
        cells: dict[tuple[int, int], None] = {}
        for row, col, _ in self.stones:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = row + dr, col + dc
                    if in_bounds(r, c) and self.grid[r][c] == EMPTY:
                        cells[(r, c)] = None
        found = sorted(cells)
        if radius == 1:
            self._neighbour_cache = found
        return found
