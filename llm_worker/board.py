"""The board of the game, as text and as coordinates.

The module keeps the drawing of the board and the reading of a coordinate
apart from the model client, so the engine can be tested without a network.
"""

from __future__ import annotations

import re

# The size of the board and the labels of its lines.
BOARD_SIZE = 15
COLUMN_LABELS = "ABCDEFGHIJKLMNO"

# The marks of the cells in the drawing.
EMPTY_MARK = "."
BLACK_MARK = "X"
WHITE_MARK = "O"

# The colors of the players.
BLACK = 1
WHITE = 2

# A coordinate of the board, like "H8".
COORDINATE_PATTERN = re.compile(r"([A-Oa-o])\s*[-]?\s*(\d{1,2})")


def player_name(player: int) -> str:
    """Return the name of a player."""
    return "黑棋" if player == BLACK else "白棋"


def render_board(board: list[list[int]]) -> str:
    """Draw the board with its coordinate labels.

    The columns are the letters A to O and the rows the numbers 1 to 15, the
    same labels the desktop interface shows, so a coordinate of the answer
    points at the cell the user sees.
    """
    header = "    " + " ".join(COLUMN_LABELS)
    lines = [header]
    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            value = board[row][col] if row < len(board) and col < len(board[row]) else 0
            if value == BLACK:
                mark = BLACK_MARK
            elif value == WHITE:
                mark = WHITE_MARK
            else:
                mark = EMPTY_MARK
            cells.append(mark)
        lines.append(f"{row + 1:02d} " + " ".join(cells))
    return "\n".join(lines)


def render_history(history: list[dict]) -> str:
    """Draw the moves of the game in order."""
    if not history:
        return "（暂无）"
    moves = []
    for move in history:
        coordinate = coordinate_of(move["row"], move["col"])
        moves.append(f"{move['number']}. {player_name(move['player'])} {coordinate}")
    return "\n".join(moves)


def coordinate_of(row: int, col: int) -> str:
    """Return the label of a cell, like "H8"."""
    if not in_bounds(row, col):
        return "?"
    return f"{COLUMN_LABELS[col]}{row + 1}"


def in_bounds(row: int, col: int) -> bool:
    """Report whether a cell is on the board."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_empty(board: list[list[int]], row: int, col: int) -> bool:
    """Report whether a cell holds no stone."""
    if not in_bounds(row, col):
        return False
    return board[row][col] == 0


def parse_coordinate(text: str) -> tuple[int, int] | None:
    """Read a coordinate like "H8" and return the row and the column.

    The row and the column of the answer are the ones of the board labels,
    so the first row is 1 and the first column is A. None is returned when
    the text holds no coordinate.
    """
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    col = COLUMN_LABELS.index(match.group(1).upper())
    row = int(match.group(2)) - 1
    if not in_bounds(row, col):
        return None
    return row, col


def parse_move(text: str) -> tuple[int, int] | None:
    """Read a move from the answer of the model.

    The answer is a JSON object with the row and the column of the cell, in
    the 0-based numbers of the API, or a coordinate of the board labels.
    """
    row, col = _parse_json_move(text)
    if row is not None and col is not None:
        return row, col
    return parse_coordinate(text)


def _parse_json_move(text: str) -> tuple[int | None, int | None]:
    """Read the row and the column of a JSON answer."""
    import json

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        return None, None
    return row, col


def winning_line(board: list[list[int]], player: int) -> list[tuple[int, int]] | None:
    """Return the five cells in a row of the player, when there is one."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != player:
                continue
            for dr, dc in directions:
                line = [(row, col)]
                r, c = row + dr, col + dc
                while in_bounds(r, c) and board[r][c] == player:
                    line.append((r, c))
                    r, c = r + dr, c + dc
                if len(line) >= 5:
                    return line[:5]
    return None
