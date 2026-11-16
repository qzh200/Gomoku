from __future__ import annotations

import json
import re


BOARD_SIZE = 15
COLUMN_LABELS = "ABCDEFGHIJKLMNO"


EMPTY_MARK = "."
BLACK_MARK = "X"
WHITE_MARK = "O"


BLACK = 1
WHITE = 2


COORDINATE_PATTERN = re.compile(r"([A-Oa-o])\s*[-]?\s*(\d{1,2})")


def player_name(player: int) -> str:
    return "黑棋" if player == BLACK else "白棋"


def render_board(board: list[list[int]]) -> str:
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
    if not history:
        return "（暂无）"
    moves = []
    for move in history:
        coordinate = coordinate_of(move["row"], move["col"])
        moves.append(f"{move['number']}. {player_name(move['player'])} {coordinate}")
    return "\n".join(moves)


def coordinate_of(row: int, col: int) -> str:
    if not in_bounds(row, col):
        return "?"
    return f"{COLUMN_LABELS[col]}{row + 1}"


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_empty(board: list[list[int]], row: int, col: int) -> bool:
    if not in_bounds(row, col):
        return False
    return board[row][col] == 0


def parse_coordinate(text: str) -> tuple[int, int] | None:
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    col = COLUMN_LABELS.index(match.group(1).upper())
    row = int(match.group(2)) - 1
    if not in_bounds(row, col):
        return None
    return row, col


def parse_move(text: str) -> tuple[int, int] | None:
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
    value = json_field(text, "move")
    if isinstance(value, str):
        return parse_coordinate(value)
    row, col = json_numbers(text)
    if row is None or col is None:
        return None
    return labels_to_cell(row, col)


def labels_to_cell(row: int, col: int) -> tuple[int, int] | None:
    if 1 <= row <= BOARD_SIZE and 1 <= col <= BOARD_SIZE:
        return row - 1, col - 1
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None


def json_field(text: str, name: str) -> object:
    payload = _json_object(text)
    if payload is None:
        return None
    return payload.get(name)


def json_numbers(text: str) -> tuple[int | None, int | None]:
    payload = _json_object(text)
    if payload is None:
        return None, None
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        return None, None
    return row, col


def _json_object(text: str) -> dict | None:
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
        if all(cell == 0 for row in board_state for cell in row):
            return [(7, 7)]
        return []
    ranked.sort()
    return [(row, col) for _, _, row, col in ranked[:limit]]


def threats(board_state: list[list[int]], player: int) -> list[str]:
    opponent = BLACK if player == WHITE else WHITE
    found: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    seen: set[tuple] = set()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != opponent:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
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
