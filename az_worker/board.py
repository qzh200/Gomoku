"""The board of the AlphaZero engine: a grid, a stack of moves and the rules.

The board keeps no score. The search plays and takes back stones tens of
thousands of times per move, so it only does the work the search needs: the
grid, the five in a row test and the candidate cells.
"""

from __future__ import annotations

# The cells of the board.
EMPTY = 0
BLACK = 1
WHITE = 2

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

# The four directions of a line: across, down, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def other(player: int) -> int:
    """Return the opponent of a player."""
    return BLACK if player == WHITE else WHITE


class Board:
    """A Gomoku board that can be played on and taken back."""

    __slots__ = ("grid", "_stones", "_moves")

    def __init__(self, rows: list[list[int]] | None = None) -> None:
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        if rows is not None:
            for row, values in enumerate(rows[:BOARD_SIZE]):
                for col, value in enumerate(values[:BOARD_SIZE]):
                    self.grid[row][col] = int(value)
        self._stones: list[tuple[int, int, int]] = []
        # The moves of the game, in the order they were played. A board that
        # was built from a snapshot does not know the order, so it has no
        # moves to take back: undo() would take the wrong stone.
        self._moves: list[tuple[int, int, int]] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                player = self.grid[row][col]
                if player != EMPTY:
                    self._stones.append((row, col, player))

    @property
    def stones(self) -> list[tuple[int, int, int]]:
        """Return the stones of the board as (row, column, player)."""
        return self._stones

    def copy(self) -> "Board":
        """Return a copy that shares nothing with this board."""
        clone = Board.__new__(Board)
        clone.grid = [row[:] for row in self.grid]
        clone._stones = self._stones[:]
        clone._moves = self._moves[:]
        return clone

    # -- playing and taking back ------------------------------------------

    def play(self, row: int, col: int, player: int) -> bool:
        """Play a stone; the return value tells whether it makes five."""
        if self.grid[row][col] != EMPTY:
            raise ValueError(f"格 ({row},{col}) 已有棋子")
        self.grid[row][col] = player
        self._stones.append((row, col, player))
        self._moves.append((row, col, player))
        return self._makes_five(row, col, player)

    def undo(self) -> None:
        """Take the last stone back."""
        if not self._moves:
            raise ValueError("没有可撤销的着法")
        row, col, _ = self._moves.pop()
        self._stones.pop()
        self.grid[row][col] = EMPTY

    # -- the rules ---------------------------------------------------------

    def _makes_five(self, row: int, col: int, player: int) -> bool:
        """Report whether the stone on this cell makes five in a row."""
        for dr, dc in DIRECTIONS:
            length = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.grid[r][c] == player:
                    length += 1
                    r += sign * dr
                    c += sign * dc
            if length >= 5:
                return True
        return False

    def wins_with(self, row: int, col: int, player: int) -> bool:
        """Report whether a stone on this cell would make five in a row."""
        if self.grid[row][col] != EMPTY:
            return False
        self.grid[row][col] = player
        try:
            return self._makes_five(row, col, player)
        finally:
            self.grid[row][col] = EMPTY

    def is_full(self) -> bool:
        """Report whether no cell is free."""
        return len(self._stones) >= BOARD_CELLS

    def move_count(self) -> int:
        """Return how many stones are on the board."""
        return len(self._stones)

    # -- the moves worth looking at ---------------------------------------

    def candidates(self, radius: int = 2) -> list[tuple[int, int]]:
        """Return the empty cells within ``radius`` of a stone, sorted."""
        if not self._stones:
            centre = BOARD_SIZE // 2
            return [(centre, centre)]
        cells: dict[tuple[int, int], None] = {}
        for row, col, _ in self._stones:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.grid[r][c] == EMPTY:
                        cells[(r, c)] = None
        return sorted(cells)
