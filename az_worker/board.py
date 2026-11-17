from __future__ import annotations


EMPTY = 0
BLACK = 1
WHITE = 2

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


class Board:
    __slots__ = ("grid", "_stones")

    def __init__(self, rows: list[list[int]] | None = None) -> None:
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        if rows is not None:
            for row, values in enumerate(rows[:BOARD_SIZE]):
                for col, value in enumerate(values[:BOARD_SIZE]):
                    self.grid[row][col] = int(value)
        self._stones: list[tuple[int, int, int]] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                player = self.grid[row][col]
                if player != EMPTY:
                    self._stones.append((row, col, player))

    @property
    def stones(self) -> list[tuple[int, int, int]]:
        return self._stones

    def copy(self) -> "Board":
        clone = Board.__new__(Board)
        clone.grid = [row[:] for row in self.grid]
        clone._stones = self._stones[:]
        return clone

    def play(self, row: int, col: int, player: int) -> bool:
        if self.grid[row][col] != EMPTY:
            raise ValueError(f"格 ({row},{col}) 已有棋子")
        self.grid[row][col] = player
        self._stones.append((row, col, player))
        return self._makes_five(row, col, player)


    def _makes_five(self, row: int, col: int, player: int) -> bool:
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
        if self.grid[row][col] != EMPTY:
            return False
        self.grid[row][col] = player
        try:
            return self._makes_five(row, col, player)
        finally:
            self.grid[row][col] = EMPTY

    def is_full(self) -> bool:
        return len(self._stones) >= BOARD_CELLS

    def candidates(self, radius: int = 2) -> list[tuple[int, int]]:
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
