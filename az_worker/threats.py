from __future__ import annotations

from .board import BOARD_SIZE


STRIDE = BOARD_SIZE + 5

TOTAL_BITS = STRIDE * BOARD_SIZE

ALL_BITS = (1 << TOTAL_BITS) - 1

VALID_MASK = 0
for _row in range(BOARD_SIZE):
    for _col in range(BOARD_SIZE):
        VALID_MASK |= 1 << (_row * STRIDE + _col)

DIRECTIONS = (1, STRIDE, STRIDE + 1, STRIDE - 1)

NEIGHBOUR_SHIFTS = (1, -1, STRIDE, -STRIDE, STRIDE - 1, 1 - STRIDE, STRIDE + 1, -STRIDE - 1)


def board_masks(board, player: int) -> tuple[int, int, int]:
    own = 0
    other = 0
    for row, col, stone in board.stones:
        bit = 1 << (row * STRIDE + col)
        if stone == player:
            own |= bit
        else:
            other |= bit
    return own, other, VALID_MASK & ~(own | other)


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _products(values: list[int]) -> tuple[list[int], list[int]]:
    count = len(values)
    prefix = [ALL_BITS] * (count + 1)
    suffix = [ALL_BITS] * (count + 1)
    for index in range(count):
        prefix[index + 1] = prefix[index] & values[index]
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] & values[index]
    return prefix, suffix


def five_mask(stones: int, empty: int) -> int:
    total = 0
    for shift in DIRECTIONS:
        gaps = [empty >> (j * shift) for j in range(5)]
        line = [stones >> (j * shift) for j in range(5)]
        prefix, suffix = _products(line)
        for hole in range(5):
            total |= (gaps[hole] & prefix[hole] & suffix[hole + 1]) << (hole * shift)
    return total & empty


def open_four_mask(stones: int, empty: int) -> int:
    total = 0
    for shift in DIRECTIONS:
        gaps = [empty >> (j * shift) for j in range(6)]
        ends = gaps[0] & gaps[5]
        line = [stones >> (j * shift) for j in range(5)]
        middle_prefix, middle_suffix = _products(line[1:5])
        for hole in range(4):
            others = middle_prefix[hole] & middle_suffix[hole + 1]
            total |= (ends & gaps[hole + 1] & others) << ((hole + 1) * shift)
    return total & empty


def open_three_mask(stones: int, empty: int) -> int:
    total = 0
    for shift in DIRECTIONS:
        for start in range(5):
            outer = (empty >> (start * shift)) & (empty >> ((start + 4) * shift))
            if not outer:
                continue
            middle = [stones >> ((start + index) * shift) for index in (1, 2, 3)]
            holes = [empty >> ((start + index) * shift) for index in (1, 2, 3)]
            prefix, suffix = _products(middle)
            for hole in range(3):
                total |= (
                    outer & holes[hole] & prefix[hole] & suffix[hole + 1]
                ) << ((start + hole + 1) * shift)
    return total & empty


def threat_masks(own: int, other: int, empty: int) -> tuple[int, int, int, int, int, int]:
    return (
        five_mask(own, empty),
        five_mask(other, empty),
        open_four_mask(own, empty),
        open_four_mask(other, empty),
        open_three_mask(own, empty),
        open_three_mask(other, empty),
    )


def near_mask(stones: int, empty: int) -> int:
    near = 0
    for shift in NEIGHBOUR_SHIFTS:
        near |= stones << shift if shift > 0 else stones >> -shift
    return near & VALID_MASK & empty
