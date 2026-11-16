from __future__ import annotations

from .board import BOARD_SIZE


PLANE_COUNT = 3


PLANE_SELF = 0

PLANE_OTHER = 1

PLANE_LAST = 2


def _transforms() -> list[tuple[int, int]]:
    def identity(cell: int) -> int:
        return cell

    def rot90(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return col * BOARD_SIZE + (BOARD_SIZE - 1 - row)

    def rot180(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return (BOARD_SIZE - 1 - row) * BOARD_SIZE + (BOARD_SIZE - 1 - col)

    def rot270(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return (BOARD_SIZE - 1 - col) * BOARD_SIZE + row

    def flip(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return row * BOARD_SIZE + (BOARD_SIZE - 1 - col)

    def flip_rot90(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return (BOARD_SIZE - 1 - col) * BOARD_SIZE + (BOARD_SIZE - 1 - row)

    def flip_rot180(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return (BOARD_SIZE - 1 - row) * BOARD_SIZE + col

    def flip_rot270(cell: int) -> int:
        row, col = divmod(cell, BOARD_SIZE)
        return col * BOARD_SIZE + row

    forwards = (identity, rot90, rot180, rot270, flip, flip_rot90, flip_rot180, flip_rot270)
    backwards = []
    for forward in forwards:
        inverse = list(range(BOARD_SIZE * BOARD_SIZE))
        for cell in range(BOARD_SIZE * BOARD_SIZE):
            inverse[forward(cell)] = cell
        backwards.append(inverse)
    return list(zip(forwards, backwards))


TRANSFORMS = _transforms()


def encode(board, player: int, last_move: tuple[int, int] | None) -> list[list[list[float]]]:
    planes = [
        [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)],
    ]
    for row, col, stone in board.stones:
        plane = PLANE_SELF if stone == player else PLANE_OTHER
        planes[plane][row][col] = 1.0
    if last_move is not None:
        planes[PLANE_LAST][last_move[0]][last_move[1]] = 1.0
    return planes


def transform_planes(planes, index: int) -> list[list[list[float]]]:
    forward, _ = TRANSFORMS[index]
    result = []
    for plane in planes:
        new_plane = [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if plane[row][col]:
                    new_row, new_col = divmod(forward(row * BOARD_SIZE + col), BOARD_SIZE)
                    new_plane[new_row][new_col] = plane[row][col]
        result.append(new_plane)
    return result


def transform_policy(policy: list[float], index: int) -> list[float]:
    forward, _ = TRANSFORMS[index]
    result = [0.0] * len(policy)
    for cell, value in enumerate(policy):
        result[forward(cell)] = value
    return result


def legal_mask(board, radius: int = 1) -> list[float]:
    mask = [0.0] * (BOARD_SIZE * BOARD_SIZE)
    for row, col in board.candidates(radius):
        mask[row * BOARD_SIZE + col] = 1.0
    return mask
