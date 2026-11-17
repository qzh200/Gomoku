from __future__ import annotations

import json
from typing import NamedTuple


BOARD_SIZE = 15


BLACK = 1
WHITE = 2


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


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


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_empty(board: list[list[int]], row: int, col: int) -> bool:
    if not in_bounds(row, col):
        return False
    return board[row][col] == 0


def wins_with(board_state: list[list[int]], player: int, row: int, col: int) -> bool:
    """假设 player 落在 (row, col)，这一手是否连成五子。"""
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            r, c = row + dr * sign, col + dc * sign
            while in_bounds(r, c) and board_state[r][c] == player:
                count += 1
                r, c = r + dr * sign, c + dc * sign
        if count >= 5:
            return True
    return False


def five_points(board_state: list[list[int]], player: int) -> list[tuple[int, int]]:
    """player 下在这些空点就成五。"""
    return [
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if board_state[row][col] == 0 and wins_with(board_state, player, row, col)
    ]


def four_makers(
    board_state: list[list[int]], player: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """落子后自己有几个成五点：两个以上是活四，一个就是冲四。

    只检查已有棋子两格以内的空点，别处的落子连不成四。
    """
    open_fours: list[tuple[int, int]] = []
    fours: list[tuple[int, int]] = []
    for row, col in _candidates(board_state):
        board_state[row][col] = player
        try:
            gained = _new_five_points(board_state, player, row, col)
        finally:
            board_state[row][col] = 0
        if len(gained) >= 2:
            open_fours.append((row, col))
        elif len(gained) == 1:
            fours.append((row, col))
    return open_fours, fours


class Threats(NamedTuple):
    """一方的成五点、活四点、冲四点、双活三点。越靠前越强。"""

    fives: list[tuple[int, int]]
    open_fours: list[tuple[int, int]]
    fours: list[tuple[int, int]]
    doubles: list[tuple[int, int]]


def threats(board_state: list[list[int]], player: int) -> Threats:
    fives = five_points(board_state, player)
    if fives:
        return Threats(fives, [], [], [])
    open_fours, fours = four_makers(board_state, player)
    if open_fours:
        return Threats([], open_fours, fours, [])
    return Threats([], [], fours, double_threes(board_state, player))


def describe(found: Threats, player: int) -> list[str]:
    name = player_name(player)

    if len(found.fives) >= 2:
        return [
            f"{name}有 {len(found.fives)} 个成五点 {spots(found.fives)}，落一子就成五，堵一个也挡不住。"
        ]
    if found.fives:
        return [f"{name}下在 {spots(found.fives)} 就成五。"]

    lines: list[str] = []
    if found.open_fours:
        lines.append(
            f"{name}下在 {spots(found.open_fours)} 就成活四：之后有两个成五点，挡不住。"
        )
    if found.doubles:
        lines.append(
            f"{name}下在 {spots(found.doubles)} 会同时做出两个活三：对手只能挡一边，"
            "另一边下一手就成活四。"
        )
    if found.fours:
        lines.append(
            f"{name}下在 {spots(found.fours, limit=4)} 就成冲四：只有一个成五点，对手必须去堵。"
        )
    return lines


def double_threes(board_state: list[list[int]], player: int) -> list[tuple[int, int]]:
    """落子后同时有两个以上活四点：双活三，对手挡不住。"""
    found: list[tuple[int, int]] = []
    for row, col in _candidates(board_state):
        board_state[row][col] = player
        try:
            if _new_five_points(board_state, player, row, col):
                continue
            makers = _open_four_makers_near(board_state, player, row, col)
        finally:
            board_state[row][col] = 0
        if len(makers) >= 2:
            found.append((row, col))
    return found


def _open_four_makers_near(
    board_state: list[list[int]], player: int, row: int, col: int
) -> list[tuple[int, int]]:
    """刚落下的这一子造出哪些活四点。活四点必然和这一子在同一线上。"""
    found: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        for step in range(1, 5):
            for sign in (-1, 1):
                r, c = row + dr * step * sign, col + dc * step * sign
                if not in_bounds(r, c) or board_state[r][c] != 0:
                    continue
                if (r, c) in found:
                    continue
                board_state[r][c] = player
                try:
                    gained = _new_five_points(board_state, player, r, c)
                finally:
                    board_state[r][c] = 0
                if len(gained) >= 2:
                    found.append((r, c))
    return found


def spots(cells: list[tuple[int, int]], limit: int | None = None) -> str:
    shown = cells if limit is None or len(cells) <= limit else cells[:limit]
    text = "、".join(f"({row}, {col})" for row, col in shown)
    if len(shown) < len(cells):
        text += f" 等 {len(cells)} 个点"
    return text


def _candidates(board_state: list[list[int]]) -> list[tuple[int, int]]:
    cells = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != 0:
                continue
            if _has_stone_near(board_state, row, col):
                cells.append((row, col))
    return cells


def _has_stone_near(board_state: list[list[int]], row: int, col: int) -> bool:
    for dr in (-2, -1, 0, 1, 2):
        for dc in (-2, -1, 0, 1, 2):
            r, c = row + dr, col + dc
            if in_bounds(r, c) and board_state[r][c] != 0:
                return True
    return False


def _new_five_points(
    board_state: list[list[int]], player: int, row: int, col: int
) -> list[tuple[int, int]]:
    """刚落下的这一子新造出哪些成五点。新的一线必然经过这一子。"""
    found: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        for step in range(1, 5):
            for sign in (-1, 1):
                r, c = row + dr * step * sign, col + dc * step * sign
                if not in_bounds(r, c) or board_state[r][c] != 0:
                    continue
                if (r, c) in found:
                    continue
                if wins_with(board_state, player, r, c):
                    found.append((r, c))
    return found


def parse_move(text: str) -> tuple[int, int] | None:
    payload = _json_object(text)
    if payload is None:
        return None
    value = payload.get("move")
    if isinstance(value, dict):
        payload = value
    row = payload.get("row")
    col = payload.get("col")
    if isinstance(row, int) and isinstance(col, int) and in_bounds(row, col):
        return row, col
    return None


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
