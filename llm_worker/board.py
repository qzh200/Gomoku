"""The board of the game, as text and as coordinates.

The module keeps the drawing of the board and the reading of a coordinate
apart from the model client, so the engine can be tested without a network.
"""

from __future__ import annotations

import json
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

    The board the model reads is labelled with its coordinates, so the answer
    is expected in that same notation: a JSON object with a ``move`` field
    holding a coordinate like ``M5``, or the bare coordinate. Numbers are
    read in the labels of the board too, never as 0-based indexes, because a
    model answers what it sees and an off-by-one would move the stone to a
    cell nobody asked for.
    """
    value = json_field(text, "move")
    if isinstance(value, str):
        cell = parse_coordinate(value)
        if cell is not None:
            return cell
    cell = parse_coordinate(text)
    if cell is not None:
        return cell
    row, col = json_numbers(text)
    if row is None or col is None:
        return None
    return labels_to_cell(row, col)


def parse_move_strict(text: str) -> tuple[int, int] | None:
    """Read a move from an answer that must be a JSON object.

    The reasoning text of a model mentions coordinates in passing, so only an
    explicit object is trusted there.
    """
    value = json_field(text, "move")
    if isinstance(value, str):
        return parse_coordinate(value)
    row, col = json_numbers(text)
    if row is None or col is None:
        return None
    return labels_to_cell(row, col)


def labels_to_cell(row: int, col: int) -> tuple[int, int] | None:
    """Turn the labels of the board into the 0-based cell of the API.

    The labels count the rows from 1 and the columns from A, so ``M5`` is the
    row 5 and the column 13. A number pair outside that range is read as the
    0-based indexes of the API instead, which is what a model that ignores
    the labels answers.
    """
    if 1 <= row <= BOARD_SIZE and 1 <= col <= BOARD_SIZE:
        return row - 1, col - 1
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None


def json_field(text: str, name: str) -> object:
    """Return one field of the JSON object in the text, or None."""
    payload = _json_object(text)
    if payload is None:
        return None
    return payload.get(name)


def json_numbers(text: str) -> tuple[int | None, int | None]:
    """Read the row and the column of a JSON answer."""
    payload = _json_object(text)
    if payload is None:
        return None, None
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        return None, None
    return row, col


def _json_object(text: str) -> dict | None:
    """Return the JSON object in the text, or None.

    The answer may carry prose around the object, so the object is taken out
    of the text first.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def candidates(board_state: list[list[int]], limit: int = 40) -> list[tuple[int, int]]:
    """Return the empty cells worth playing, nearest to the stones first.

    A model reading a 15x15 grid of dots easily names a cell that already
    holds a stone. Handing it the free cells takes that mistake away, while
    the choice stays its own. Empty cells next to a stone come first, then
    those one line further out; on an empty board the center is the move.
    """
    ranked: list[tuple[int, int, int, int]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != 0:
                continue
            nearest = 3
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = row + dr, col + dc
                    if in_bounds(r, c) and board_state[r][c] != 0:
                        distance = max(abs(dr), abs(dc))
                        if distance < nearest:
                            nearest = distance
            if nearest <= 2:
                ranked.append((nearest, abs(row - 7) + abs(col - 7), row, col))
    if not ranked:
        return [(7, 7)]
    ranked.sort()
    return [(row, col) for _, _, row, col in ranked[:limit]]


def threats(board_state: list[list[int]], player: int) -> list[str]:
    """Describe the lines of the opponent that this player has to answer.

    A model reading a grid of dots does not count runs reliably, so the
    backend counts them and names the cells that block: a four that cannot be
    blocked loses the game at once, a three is the next thing to stop. The
    choice stays with the model; only the counting moves here.
    """
    opponent = BLACK if player == WHITE else WHITE
    found: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    seen: set[tuple] = set()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != opponent:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                # Only start at the first stone of a run.
                if in_bounds(row - dr, col - dc) and board_state[row - dr][col - dc] == opponent:
                    continue
                run = []
                r, c = row, col
                while in_bounds(r, c) and board_state[r][c] == opponent:
                    run.append((r, c))
                    r, c = r + dr, c + dc
                if len(run) < 3:
                    continue
                ends = []
                for er, ec in ((row - dr, col - dc), (r, c)):
                    if in_bounds(er, ec) and board_state[er][ec] == 0:
                        ends.append((er, ec))
                if not ends:
                    continue
                key = (tuple(run), tuple(ends))
                if key in seen:
                    continue
                seen.add(key)
                found.append((len(run), run, ends))
    found.sort(key=lambda item: item[0], reverse=True)
    lines = []
    for length, run, ends in found[:3]:
        cells = "/".join(coordinate_of(r, c) for r, c in run)
        spots = "、".join(coordinate_of(r, c) for r, c in ends)
        lines.append(f"对手在 {cells} 有 {length} 连，能挡住的点：{spots}")
    return lines


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
