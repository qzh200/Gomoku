"""The features of a position: three planes of 15 by 15.

The network always looks at a position from the side of the player to move,
so the two stone planes are swapped when the colour changes. The third plane
marks the cell of the last move, which tells the network where the game
continues.

The eight symmetries of the square board are computed from the same table, so
the training data can be mirrored and rotated without another implementation.
"""

from __future__ import annotations

from .board import BOARD_SIZE

# The number of planes the network takes.
PLANE_COUNT = 3

# The plane of the stones of the player to move.
PLANE_SELF = 0
# The plane of the stones of the opponent.
PLANE_OTHER = 1
# The plane of the last move.
PLANE_LAST = 2


def _transforms() -> list[tuple[int, int]]:
    """Return the eight symmetries of the square board.

    Every transform is a pair of functions: one maps a cell to its image and
    one maps an image back to the cell it came from.
    """

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
        # The rotation of the flipped board, so the eight transforms are the
        # eight symmetries of the square and no two of them are the same.
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
    """Return the planes of a position.

    The result is a list of three planes; every plane is a list of rows and
    every row a list of zeros and ones.
    """
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
    """Return the planes under one of the eight symmetries."""
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
    """Return a policy vector under one of the eight symmetries.

    The vector is over the cells of the board, so the entry of a cell moves
    to the cell its image sits on.
    """
    forward, _ = TRANSFORMS[index]
    result = [0.0] * len(policy)
    for cell, value in enumerate(policy):
        result[forward(cell)] = value
    return result


def legal_mask(board, radius: int = 1) -> list[float]:
    """Return a mask that is one on every cell a move may be played on."""
    mask = [0.0] * (BOARD_SIZE * BOARD_SIZE)
    for row, col in board.candidates(radius):
        mask[row * BOARD_SIZE + col] = 1.0
    return mask
