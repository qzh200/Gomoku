from __future__ import annotations


EMPTY = 0
BLACK = 1
WHITE = 2


WINDOW = 6
POW3 = (1, 3, 9, 27, 81, 243)


OPEN_THREE_PATTERNS = ("011100", "001110", "010110", "011010")


SHAPES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("five", ("11111",), 10_000_000),
    ("open_four", ("011110",), 120_000),
    ("four", ("011112", "211110", "11011", "10111", "11101"), 15_000),
    ("open_three", OPEN_THREE_PATTERNS, 6_000),
    ("closed_three", ("001112", "211100"), 600),
    ("open_two", ("001100", "010100"), 300),
    ("closed_two", ("001120", "021100"), 50),
    ("one", ("001000",), 2),
)


def decode(code: int, length: int = WINDOW) -> list[int]:
    cells = []
    for _ in range(length):
        cells.append(code % 3)
        code //= 3
    return cells


def classify(cells: list[int] | tuple[int, ...]) -> tuple[str, int]:
    text = "".join(str(value) for value in cells)
    for name, patterns, score in SHAPES:
        for pattern in patterns:
            if pattern in text:
                return name, score
    return "none", 0


def _swapped(cells: list[int] | tuple[int, ...]) -> list[int]:
    return [0 if value == EMPTY else 3 - value for value in cells]


def _build_tables() -> tuple[tuple[int, int], ...]:
    table = []
    for code in range(3**WINDOW):
        cells = decode(code)
        table.append((classify(cells)[1], classify(_swapped(cells))[1]))
    return tuple(table)


WINDOW_SCORES = _build_tables()
