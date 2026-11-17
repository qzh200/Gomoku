from __future__ import annotations

import json
import re


BOARD_SIZE = 15
COLUMN_LABELS = "ABCDEFGHIJKLMNO"


BLACK = 1
WHITE = 2


COORDINATE_PATTERN = re.compile(r"([A-Oa-o])\s*[-]?\s*(\d{1,2})")


def player_name(player: int) -> str:
    return "黑棋" if player == BLACK else "白棋"


def opponent_of(player: int) -> int:
    return BLACK if player == WHITE else WHITE


def stone_list(board_state: list[list[int]], player: int) -> str:
    """把一方的棋子写成 (行, 列) 列表，行号和列号都从 0 数起。"""
    cells = [
        f"({row}, {col})"
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if row < len(board_state)
        and col < len(board_state[row])
        and board_state[row][col] == player
    ]
    return " ".join(cells) if cells else "（没有棋子）"


def connected_lines(
    board_state: list[list[int]], player: int, minimum: int = 3
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """找出连成一线的棋子，以及两端的空点。"""
    found: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    seen: set[tuple] = set()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if row >= len(board_state) or col >= len(board_state[row]):
                continue
            if board_state[row][col] != player:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if in_bounds(row - dr, col - dc) and board_state[row - dr][col - dc] == player:
                    continue
                run = []
                r, c = row, col
                while in_bounds(r, c) and board_state[r][c] == player:
                    run.append((r, c))
                    r, c = r + dr, c + dc
                if len(run) < minimum:
                    continue
                key = tuple(run)
                if key in seen:
                    continue
                seen.add(key)
                free = []
                for er, ec in ((row - dr, col - dc), (r, c)):
                    if in_bounds(er, ec) and board_state[er][ec] == 0:
                        free.append((er, ec))
                found.append((run, free))
    return found


def describe_lines(board_state: list[list[int]], player: int, minimum: int = 3) -> str:
    """把连成一线的棋子写成文字，模型才读得出斜线。"""
    parts = []
    for run, free in connected_lines(board_state, player, minimum):
        spots = "、".join(f"({r}, {c})" for r, c in free) if free else "两端都被堵住"
        cells = "-".join(f"({rr}, {cc})" for rr, cc in run)
        parts.append(f"{cells}（{len(run)} 连，延伸端：{spots}）")
    if not parts:
        return f"{player_name(player)}没有连成三子以上的棋子。"
    return f"{player_name(player)}连成一线的地方：" + "；".join(parts) + "。"


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_empty(board: list[list[int]], row: int, col: int) -> bool:
    if not in_bounds(row, col):
        return False
    return board[row][col] == 0


def parse_coordinate(text: str) -> tuple[int, int] | None:
    """兼容模型用 H8 这种坐标标注作答的情况。"""
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    col = COLUMN_LABELS.index(match.group(1).upper())
    row = int(match.group(2)) - 1
    if not in_bounds(row, col):
        return None
    return row, col


def parse_move(text: str) -> tuple[int, int] | None:
    cell = _json_cell(text)
    if cell is not None:
        return cell
    return parse_coordinate(text)


def parse_move_strict(text: str) -> tuple[int, int] | None:
    return _json_cell(text)


def _json_cell(text: str) -> tuple[int, int] | None:
    payload = _json_object(text)
    if payload is None:
        return None
    value = payload.get("move")
    if isinstance(value, dict):
        return dict_to_cell(value)
    if isinstance(value, str):
        return parse_coordinate(value)
    row = payload.get("row")
    col = payload.get("col")
    if isinstance(row, int) and isinstance(col, int) and in_bounds(row, col):
        return row, col
    return None


def dict_to_cell(payload: dict) -> tuple[int, int] | None:
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    if not in_bounds(row, col):
        return None
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


def fallback_cell(board_state: list[list[int]]) -> tuple[int, int] | None:
    """模型不配合时兜底：优先挨着棋子的空点，其次离中心最近的空点。"""
    best: tuple[int, int] | None = None
    best_score: tuple[int, int] | None = None
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != 0:
                continue
            near = any(
                in_bounds(row + dr, col + dc)
                and board_state[row + dr][col + dc] != 0
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
            )
            score = (0 if near else 1, abs(row - 7) + abs(col - 7))
            if best_score is None or score < best_score:
                best_score = score
                best = (row, col)
    return best
