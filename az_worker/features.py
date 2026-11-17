from __future__ import annotations

import numpy as np

from .board import BOARD_SIZE
from .threats import STRIDE, board_masks, threat_masks


PLANE_COUNT = 9


PLANE_SELF = 0

PLANE_OTHER = 1

PLANE_LAST = 2

PLANE_SELF_FIVE = 3

PLANE_OTHER_FIVE = 4

PLANE_SELF_OPEN_FOUR = 5

PLANE_OTHER_OPEN_FOUR = 6

PLANE_SELF_OPEN_THREE = 7

PLANE_OTHER_OPEN_THREE = 8


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


CELL_PERMUTATIONS = tuple(
    np.array(backward, dtype=np.int64) for _, backward in TRANSFORMS
)


def _mark(plane: np.ndarray, mask: int) -> None:
    while mask:
        low = mask & -mask
        index = low.bit_length() - 1
        row, col = divmod(index, STRIDE)
        plane[row, col] = 1.0
        mask ^= low


def as_planes(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (PLANE_COUNT, BOARD_SIZE, BOARD_SIZE):
        found = array.shape[0] if array.ndim == 3 else array.shape
        raise ValueError(
            f"平面数是 {found}，现在是 {PLANE_COUNT}，这份数据过期了，要重新采集"
        )
    return array


def encode(board, player: int, last_move: tuple[int, int] | None) -> np.ndarray:
    planes = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row, col, stone in board.stones:
        planes[PLANE_SELF if stone == player else PLANE_OTHER, row, col] = 1.0
    if last_move is not None:
        planes[PLANE_LAST, last_move[0], last_move[1]] = 1.0

    own, other, empty = board_masks(board, player)
    masks = threat_masks(own, other, empty)
    _mark(planes[PLANE_SELF_FIVE], masks[0])
    _mark(planes[PLANE_OTHER_FIVE], masks[1])
    _mark(planes[PLANE_SELF_OPEN_FOUR], masks[2])
    _mark(planes[PLANE_OTHER_OPEN_FOUR], masks[3])
    _mark(planes[PLANE_SELF_OPEN_THREE], masks[4])
    _mark(planes[PLANE_OTHER_OPEN_THREE], masks[5])
    return planes


def legal_mask(board) -> np.ndarray:
    mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    for row, col in board.empty_cells():
        mask[row * BOARD_SIZE + col] = 1.0
    return mask
