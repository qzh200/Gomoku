"""The position of a Gomoku game and its static score.

The position keeps the score of both players up to date while stones are
played and taken back, so the search reads a number instead of scanning the
board. The score comes from the runs of stones: a run of five is the game, an
open four nearly is, an open three is worth a search.

An *open end* is an empty cell next to a run; a run with none is dead and
scores zero.
"""

from __future__ import annotations

BOARD_SIZE = 15
COLUMN_LABELS = "ABCDEFGHIJKLMNO"

# The cells of the board.
EMPTY = 0
BLACK = 1
WHITE = 2

# The four directions of a line: across, down, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# The score of a finished line, above every other value.
WIN_SCORE = 10_000_000

# The value of one run, by its length and by how many of its ends are open.
RUN_SCORES = {
    1: {1: 1, 2: 2},
    2: {1: 50, 2: 200},
    3: {1: 500, 2: 5_000},
    4: {1: 10_000, 2: 100_000},
}


def other(player: int) -> int:
    """Return the opponent of a player."""
    return BLACK if player == WHITE else WHITE


def run_score(length: int, open_ends: int) -> int:
    """Return the value of one run of stones."""
    if length >= 5:
        return WIN_SCORE
    return RUN_SCORES.get(length, {}).get(open_ends, 0)


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

    __slots__ = ("row", "col", "player", "gain", "win")

    def __init__(self, row: int, col: int, player: int, gain: int, win: bool) -> None:
        self.row = row
        self.col = col
        self.player = player
        self.gain = gain
        self.win = win


class Position:
    """A board with the score of both players."""

    def __init__(self, rows: list[list[int]] | None = None) -> None:
        if rows is None:
            self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        else:
            self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            for row, values in enumerate(rows[:BOARD_SIZE]):
                for col, value in enumerate(values[:BOARD_SIZE]):
                    self.grid[row][col] = int(value)
        self.totals = {BLACK: 0, WHITE: 0}
        # The stones in the order they were played, so the neighbours of a
        # stone are found without scanning the board.
        self.stones: list[tuple[int, int, int]] = []
        self._index()

    def copy(self) -> "Position":
        """Return a copy that shares nothing with this position."""
        return Position(self.grid)

    # -- the score of the board ------------------------------------------

    def _index(self) -> None:
        """Build the totals and the list of stones from the grid."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                player = self.grid[row][col]
                if player == EMPTY:
                    continue
                self.stones.append((row, col, player))
                for dr, dc in DIRECTIONS:
                    if in_bounds(row - dr, col - dc) and self.grid[row - dr][col - dc] == player:
                        continue
                    length, opens = self._run_at(row, col, dr, dc, player)
                    self.totals[player] += run_score(length, opens)

    def _run_at(self, row: int, col: int, dr: int, dc: int, player: int) -> tuple[int, int]:
        """Return the length of the run at a stone and its open ends."""
        length = 0
        r, c = row, col
        while in_bounds(r, c) and self.grid[r][c] == player:
            length += 1
            r, c = r + dr, c + dc
        opens = 1 if in_bounds(r, c) and self.grid[r][c] == EMPTY else 0
        br, bc = row - dr, col - dc
        if in_bounds(br, bc) and self.grid[br][bc] == EMPTY:
            opens += 1
        return length, opens

    def evaluate(self, player: int) -> int:
        """Return the score of the board for one player."""
        return self.totals[player] - self.totals[other(player)]

    # -- playing and taking back ------------------------------------------

    def gain_of(self, row: int, col: int, player: int) -> int:
        """Return how much the score of a player grows by playing a cell.

        The stone may join two runs in a direction, so the two runs are
        replaced by the one they become.
        """
        gain = 0
        for dr, dc in DIRECTIONS:
            left = self._span(row, col, -dr, -dc, player)
            right = self._span(row, col, dr, dc, player)
            left_open = 1 if self._is_empty(left[1], left[2]) else 0
            right_open = 1 if self._is_empty(right[1], right[2]) else 0
            # The cell that is about to be played is an open end of both runs.
            before = run_score(left[0], left_open + 1) + run_score(right[0], right_open + 1)
            after = run_score(left[0] + 1 + right[0], left_open + right_open)
            gain += after - before
        return gain

    def _span(self, row: int, col: int, dr: int, dc: int, player: int) -> tuple[int, int, int]:
        """Return the run of a player next to a cell, and the cell past it."""
        length = 0
        r, c = row + dr, col + dc
        while in_bounds(r, c) and self.grid[r][c] == player:
            length += 1
            r, c = r + dr, c + dc
        return length, r, c

    def _is_empty(self, row: int, col: int) -> bool:
        return in_bounds(row, col) and self.grid[row][col] == EMPTY

    def wins_with(self, row: int, col: int, player: int) -> bool:
        """Report whether a stone on this cell makes five in a row."""
        for dr, dc in DIRECTIONS:
            length = 1 + self._span(row, col, dr, dc, player)[0] + self._span(row, col, -dr, -dc, player)[0]
            if length >= 5:
                return True
        return False

    def play(self, row: int, col: int, player: int) -> Undo:
        """Play a stone and return what taking it back needs."""
        if self.grid[row][col] != EMPTY:
            raise ValueError(f"格 {coordinate_of(row, col)} 已有棋子")
        gain = self.gain_of(row, col, player)
        win = self.wins_with(row, col, player)
        self.grid[row][col] = player
        self.totals[player] += gain
        self.stones.append((row, col, player))
        return Undo(row, col, player, gain, win)

    def undo(self, undo: Undo) -> None:
        """Take a stone back."""
        self.grid[undo.row][undo.col] = EMPTY
        self.totals[undo.player] -= undo.gain
        self.stones.pop()

    def is_full(self) -> bool:
        """Report whether no cell is free."""
        return len(self.stones) == BOARD_SIZE * BOARD_SIZE

    # -- the moves worth looking at ---------------------------------------

    def neighbours(self) -> list[tuple[int, int]]:
        """Return the empty cells next to a stone, without duplicates."""
        cells: dict[tuple[int, int], None] = {}
        for row, col, _ in self.stones:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = row + dr, col + dc
                    if in_bounds(r, c) and self.grid[r][c] == EMPTY:
                        cells[(r, c)] = None
        return list(cells)

    def candidates(self, player: int, limit: int) -> list[tuple[int, int]]:
        """Return the moves to consider, the greediest first.

        A move is scored by what it gains this player and, at nine tenths,
        by what it would have gained the opponent: the search then meets the
        threats before the quiet moves, and a weak move is cut away early.
        """
        if not self.stones:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        opponent = other(player)
        scored = []
        for row, col in self.neighbours():
            attack = self.gain_of(row, col, player)
            defence = self.gain_of(row, col, opponent)
            scored.append((attack * 10 + defence * 9, row, col))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(row, col) for _, row, col in scored[:limit]]
