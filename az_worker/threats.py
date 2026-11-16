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


def bit_of(row: int, col: int) -> int:
    return 1 << (row * STRIDE + col)


def board_masks(board, player: int) -> tuple[int, int, int]:
    """己方、对方、空格的位棋盘。"""
    own = 0
    other = 0
    for row, col, stone in board.stones:
        bit = 1 << (row * STRIDE + col)
        if stone == player:
            own |= bit
        else:
            other |= bit
    return own, other, VALID_MASK & ~(own | other)


def _products(values: list[int]) -> tuple[list[int], list[int]]:
    """前缀与后缀的按位与，用来一次算出“除了某个位置以外全是己方棋子”。"""
    count = len(values)
    prefix = [ALL_BITS] * (count + 1)
    suffix = [ALL_BITS] * (count + 1)
    for index in range(count):
        prefix[index + 1] = prefix[index] & values[index]
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] & values[index]
    return prefix, suffix


def threat_masks(own: int, other: int, empty: int) -> tuple[int, int, int, int]:
    """己方成五点、对方成五点、己方成活四点、对方成活四点。"""
    self_five = 0
    other_five = 0
    self_four = 0
    other_four = 0
    for shift in DIRECTIONS:
        gaps = [empty >> (j * shift) for j in range(6)]
        ends = gaps[0] & gaps[5]
        for stones, own_side in ((own, True), (other, False)):
            line = [stones >> (j * shift) for j in range(5)]
            prefix, suffix = _products(line)
            five = 0
            for hole in range(5):
                five |= (
                    gaps[hole] & prefix[hole] & suffix[hole + 1]
                ) << (hole * shift)
            middle_prefix, middle_suffix = _products(line[1:5])
            four = 0
            for hole in range(4):
                others = middle_prefix[hole] & middle_suffix[hole + 1]
                four |= (
                    ends & gaps[hole + 1] & others
                ) << ((hole + 1) * shift)
            if own_side:
                self_five |= five
                self_four |= four
            else:
                other_five |= five
                other_four |= four
    return (
        self_five & empty,
        other_five & empty,
        self_four & empty,
        other_four & empty,
    )
