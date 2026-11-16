"""The search of the AlphaZero engine: PUCT over the moves of a position.

Every node of the tree holds the moves that can be played there, how often
they were visited and how much they won. A move is picked by the PUCT
formula: the value it won so far, plus a term that grows with the visits of
the node, so a move that looks good but was barely tried gets a second look.

The leaves are evaluated in batches, so the network sees many positions at
once and the GPU is not idle between two forward passes. A leaf that waits
for the network is skipped by the selection, so a batch holds no leaf twice.

The value of a node is seen from the side of the player to move there, so a
parent reads the value of its child the other way round.
"""

from __future__ import annotations

import math
import random

from .board import BOARD_SIZE, Board, other
from .features import encode, legal_mask
from .network import ACTION_COUNT, evaluate

# The constant of the PUCT formula. It is how much a fresh move is worth
# against a move that was already tried many times.
DEFAULT_C_PUCT = 1.5

# The concentration of the Dirichlet noise at the root of the tree.
DEFAULT_DIRICHLET_ALPHA = 0.3

# The share of the noise among the priors of the root.
DEFAULT_NOISE_WEIGHT = 0.25

# The radius of the candidate cells. One means the empty cells that touch a
# stone; two reaches one cell further. One is the default because it keeps
# the visit distribution of the search sharp, and a sharp distribution is
# what the policy head learns from.
DEFAULT_CANDIDATE_RADIUS = 1


def cell_of(row: int, col: int) -> int:
    """Return the index of a cell on the board."""
    return row * BOARD_SIZE + col


def row_col_of(cell: int) -> tuple[int, int]:
    """Return the row and the column of a cell on the board."""
    return divmod(cell, BOARD_SIZE)


def apply_noise(priors: list[float], alpha: float, weight: float, rng: random.Random) -> list[float]:
    """Mix Dirichlet noise into the priors of the root."""
    noise = [rng.gammavariate(alpha, 1.0) for _ in priors]
    total = sum(noise)
    if total <= 0:
        return priors
    noise = [value / total for value in noise]
    return [
        (1.0 - weight) * prior + weight * noisy
        for prior, noisy in zip(priors, noise)
    ]


class Node:
    """One node of the search tree."""

    __slots__ = (
        "children",
        "visits",
        "value_sum",
        "virtual_visits",
        "prior",
        "board",
        "player",
        "terminal_value",
        "expanded",
        "last_move",
    )

    def __init__(self, board: Board, player: int) -> None:
        self.children: dict[int, Node] = {}
        self.visits = 0
        # The sum of the values of this position, seen from the side of the
        # player to move here.
        self.value_sum = 0.0
        # The visits a node has but was not evaluated yet; they keep the
        # selection away from a leaf that waits for the network.
        self.virtual_visits = 0
        self.prior = 0.0
        self.board = board
        self.player = player
        # The value of a finished position, seen from the side of the player
        # to move there. None while the game is still running.
        self.terminal_value: float | None = None
        self.expanded = False
        # The move that led to this node; it is the last move of the board.
        self.last_move: tuple[int, int] | None = None


class SearchResult:
    """What one search found: the moves to play and how good they look."""

    __slots__ = ("visits", "policy", "root_value", "simulations")

    def __init__(
        self,
        visits: dict[int, int],
        policy: list[float],
        root_value: float,
        simulations: int,
    ) -> None:
        self.visits = visits
        self.policy = policy
        self.root_value = root_value
        self.simulations = simulations

    def best_move(self) -> int | None:
        """Return the cell to play, or None when nothing was searched."""
        if not self.visits:
            return None
        return max(self.visits.items(), key=lambda item: item[1])[0]

    def sampled_move(self, temperature: float, rng: random.Random) -> int | None:
        """Return a move drawn from the visits, the way self-play does."""
        if not self.visits:
            return None
        if temperature <= 0:
            return self.best_move()
        moves = list(self.visits)
        weights = [self.visits[move] ** (1.0 / temperature) for move in moves]
        total = sum(weights)
        if total <= 0:
            return self.best_move()
        pick = rng.random() * total
        running = 0.0
        for move, weight in zip(moves, weights):
            running += weight
            if running >= pick:
                return move
        return self.best_move()


class MCTS:
    """Searches a position with PUCT and a network that scores the leaves."""

    def __init__(
        self,
        model,
        device,
        c_puct: float = DEFAULT_C_PUCT,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
        noise_weight: float = DEFAULT_NOISE_WEIGHT,
        batch_size: int = 32,
        candidate_radius: int = DEFAULT_CANDIDATE_RADIUS,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight
        self.batch_size = max(1, batch_size)
        self.candidate_radius = candidate_radius
        self.rng = random.Random(seed)

    # -- the search --------------------------------------------------------

    def search(
        self,
        board: Board,
        player: int,
        simulations: int = 400,
        add_noise: bool = False,
        last_move: tuple[int, int] | None = None,
    ) -> SearchResult:
        """Search a position and return the visits of its moves."""
        root = Node(board.copy(), player)
        root.last_move = last_move
        # The root is expanded before the search: its children get the
        # priors of the network.
        self._expand(root, self._evaluate([root])[0])
        if add_noise:
            self._add_root_noise(root)

        pending: list[tuple[Node, list[Node]]] = []
        waiting: set[int] = set()

        for _ in range(simulations):
            leaf, path = self._descend(root, waiting)
            if leaf is None:
                # Every child of the node waits for the network: evaluate the
                # batch now, so the search can go on.
                self._release(path)
                self._flush(pending)
                waiting.clear()
                leaf, path = self._descend(root, waiting)
                if leaf is None:
                    break
            if leaf.terminal_value is not None:
                # The game is over here; the network has nothing to say.
                self._backpropagate(path, leaf.terminal_value)
                continue
            pending.append((leaf, path))
            waiting.add(id(leaf))
            if len(pending) >= self.batch_size:
                self._flush(pending)
                waiting.clear()

        if pending:
            self._flush(pending)

        return self._collect(root)

    # -- the evaluation of the leaves --------------------------------------

    def _evaluate(self, nodes: list[Node]) -> list[list[float]]:
        """Return the policy of every node, taken from the network."""
        planes = []
        masks = []
        for node in nodes:
            planes.append(encode(node.board, node.player, node.last_move))
            masks.append(legal_mask(node.board, self.candidate_radius))
        policies, _ = evaluate(self.model, planes, masks, self.device)
        return policies

    def _flush(self, pending: list[tuple[Node, list[Node]]]) -> None:
        """Evaluate a batch of leaves, expand them and push the values up."""
        if not pending:
            return
        planes = []
        masks = []
        for leaf, _ in pending:
            planes.append(encode(leaf.board, leaf.player, leaf.last_move))
            masks.append(legal_mask(leaf.board, self.candidate_radius))
        policies, values = evaluate(self.model, planes, masks, self.device)

        for (leaf, path), policy, value in zip(pending, policies, values):
            self._expand(leaf, policy)
            self._backpropagate(path, value)
        pending.clear()

    def _expand(self, node: Node, policy: list[float]) -> None:
        """Give the node its children, one per legal move."""
        node.expanded = True
        moves = node.board.candidates(self.candidate_radius)
        if not moves:
            node.terminal_value = 0.0
            return
        priors = [policy[cell_of(row, col)] for row, col in moves]
        total = sum(priors)
        if total <= 0:
            priors = [1.0 / len(moves)] * len(moves)
        else:
            priors = [prior / total for prior in priors]
        for (row, col), prior in zip(moves, priors):
            child_board = node.board.copy()
            won = child_board.play(row, col, node.player)
            child = Node(child_board, other(node.player))
            child.prior = prior
            child.last_move = (row, col)
            if won:
                # The player to move in the child has lost the game.
                child.terminal_value = -1.0
            node.children[cell_of(row, col)] = child
        if node.board.is_full():
            node.terminal_value = 0.0

    def _add_root_noise(self, root: Node) -> None:
        """Mix the Dirichlet noise into the priors of the root."""
        moves = list(root.children)
        if not moves:
            return
        priors = [root.children[move].prior for move in moves]
        noisy = apply_noise(priors, self.dirichlet_alpha, self.noise_weight, self.rng)
        for move, prior in zip(moves, noisy):
            root.children[move].prior = prior

    # -- the walk down the tree --------------------------------------------

    def _descend(self, root: Node, waiting: set[int]) -> tuple[Node | None, list[Node]]:
        """Walk down the tree and return the leaf and the nodes on the way."""
        node = root
        path: list[Node] = [root]
        root.virtual_visits += 1
        while node.expanded and node.terminal_value is None and node.children:
            move = self._select_child(node, waiting)
            if move is None:
                return None, path
            child = node.children[move]
            child.virtual_visits += 1
            path.append(child)
            node = child
        return node, path

    def _select_child(self, node: Node, waiting: set[int]) -> int | None:
        """Return the move the PUCT formula prefers, or None when there is none."""
        sqrt_total = math.sqrt(max(node.visits, 1))
        best_move: int | None = None
        best_score = -math.inf
        for move, child in node.children.items():
            if id(child) in waiting:
                # The leaf waits for the network; it is picked another time.
                continue
            if child.visits > 0:
                # The value of the child is seen from the side of its player,
                # so the parent reads it the other way round.
                quality = -child.value_sum / child.visits
            else:
                quality = 0.0
            # The virtual visits only steer the selection away from a leaf
            # that waits for the network; they are not part of the value.
            exploration = (
                self.c_puct * child.prior * sqrt_total / (1 + child.visits + child.virtual_visits)
            )
            score = quality + exploration
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # -- the values ---------------------------------------------------------

    def _release(self, path: list[Node]) -> None:
        """Give the virtual visits of a discarded path back to the tree."""
        for node in path:
            node.virtual_visits = max(0, node.virtual_visits - 1)

    def _backpropagate(self, path: list[Node], value: float) -> None:
        """Push a value up the tree, changing the sign on every level.

        The value belongs to the leaf and is seen from the side of the player
        to move there. The player above is the other one, so the sign changes
        on every step up.
        """
        current = value
        for depth in range(len(path) - 1, -1, -1):
            node = path[depth]
            node.virtual_visits = max(0, node.virtual_visits - 1)
            node.visits += 1
            node.value_sum += current
            current = -current

    def _collect(self, root: Node) -> SearchResult:
        """Turn the tree into the visits of the moves and the value of the root."""
        visits: dict[int, int] = {}
        policy = [0.0] * ACTION_COUNT
        total = 0
        for move, child in root.children.items():
            visits[move] = child.visits
            total += child.visits
        if total > 0:
            for move, child in root.children.items():
                policy[move] = child.visits / total
        # The value of the root is seen from the side of the player to move
        # there, which is the side the value head is asked about.
        root_value = root.value_sum / root.visits if root.visits else 0.0
        return SearchResult(visits, policy, root_value, total)
