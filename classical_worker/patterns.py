"""The six-cell windows that score a board.

A board is scored by sliding a window of six cells along every line. Each
window is encoded as a base-3 number and looked up in a table built once at
import time, so scoring a window is a list index. The table knows the common
Gomoku shapes: five, open four, four, open three, closed three, open two,
closed two and one stone.

Windows overlap on purpose: one run is counted by several windows. The scores
are only compared with each other, so double counting is a calibration choice,
not a bug.
"""

from __future__ import annotations

# The cells of the board and of a window.
EMPTY = 0
BLACK = 1
WHITE = 2
BLOCKED = 2  # an opponent stone or the edge of the board

# A window is six cells long, the shortest span that holds an open four.
WINDOW = 6
POW3 = (1, 3, 9, 27, 81, 243)

# The shapes, the highest first. The first match wins, so a window that holds
# a four and a three is scored as the four.
SHAPES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("five", ("11111",), 10_000_000),
    ("open_four", ("011110",), 120_000),
    ("four", ("011112", "211110", "11011", "10111", "11101"), 15_000),
    ("open_three", ("011100", "001110", "010110", "011010"), 6_000),
    ("closed_three", ("001112", "211100"), 600),
    ("open_two", ("001100", "010100"), 300),
    ("closed_two", ("001120", "021100"), 50),
    ("one", ("001000",), 2),
)


def encode(cells: list[int] | tuple[int, ...]) -> int:
    """Return the base-3 code of a window."""
    code = 0
    for index, value in enumerate(cells):
        code += value * POW3[index]
    return code


def decode(code: int, length: int = WINDOW) -> list[int]:
    """Return the cells of a window code."""
    cells = []
    for _ in range(length):
        cells.append(code % 3)
        code //= 3
    return cells


def classify(cells: list[int] | tuple[int, ...]) -> tuple[str, int]:
    """Return the highest shape of a window and its score."""
    text = "".join(str(value) for value in cells)
    for name, patterns, score in SHAPES:
        for pattern in patterns:
            if pattern in text:
                return name, score
    return "none", 0


def _swapped(cells: list[int] | tuple[int, ...]) -> list[int]:
    """Return the window with the two colours exchanged."""
    return [0 if value == EMPTY else 3 - value for value in cells]


def _build_tables() -> tuple[tuple[int, int], ...]:
    """Return the black and white score of every window code."""
    table = []
    for code in range(3**WINDOW):
        cells = decode(code)
        table.append((classify(cells)[1], classify(_swapped(cells))[1]))
    return tuple(table)


# WINDOW_SCORES[code] = (score of black, score of white).
WINDOW_SCORES = _build_tables()


def window_score(code: int, player: int) -> int:
    """Return what a window is worth to one player."""
    return WINDOW_SCORES[code][player - 1]
