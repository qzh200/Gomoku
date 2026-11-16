from __future__ import annotations

import math
import random
import time

import numpy as np

from .board import BOARD_SIZE, Board, other
from .features import encode, legal_mask
from .network import ACTION_COUNT, evaluate


DEFAULT_C_PUCT = 1.5


DEFAULT_DIRICHLET_ALPHA = 0.3


DEFAULT_NOISE_WEIGHT = 0.25


DEFAULT_CANDIDATE_RADIUS = 1


DEFAULT_CANDIDATE_WIDTH = 16


def cell_of(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def row_col_of(cell: int) -> tuple[int, int]:
    return divmod(cell, BOARD_SIZE)


def apply_noise(priors: list[float], alpha: float, weight: float, rng: random.Random) -> list[float]:
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
    __slots__ = (
        "children",
        "visits",
        "value_sum",
        "virtual_visits",
        "prior",
        "player",
        "terminal_value",
        "expanded",
        "last_move",
        "_board",
        "_parent",
        "_move",
    )

    def __init__(
        self,
        board: Board | None,
        player: int,
        parent: "Node | None" = None,
        move: tuple[int, int] | None = None,
    ) -> None:
        self.children: dict[int, Node] = {}
        self.visits = 0


        self.value_sum = 0.0


        self.virtual_visits = 0
        self.prior = 0.0
        self.player = player


        self.terminal_value: float | None = None
        self.expanded = False

        self.last_move: tuple[int, int] | None = None
        self._board = board
        self._parent = parent
        self._move = move

    @property
    def board(self) -> Board:
        board = self._board
        if board is None:
            parent = self._parent
            move = self._move
            if parent is None or move is None:
                raise ValueError("这个节点没有棋盘")
            board = parent.board.copy()
            board.play(move[0], move[1], parent.player)
            self._board = board
        return board


class SearchResult:
    __slots__ = ("visits", "policy", "root_value", "simulations")

    def __init__(
        self,
        visits: dict[int, int],
        policy: np.ndarray,
        root_value: float,
        simulations: int,
    ) -> None:
        self.visits = visits
        self.policy = policy
        self.root_value = root_value
        self.simulations = simulations

    def best_move(self) -> int | None:
        if not self.visits:
            return None
        return max(self.visits.items(), key=lambda item: item[1])[0]

    def sampled_move(self, temperature: float, rng: random.Random) -> int | None:
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
    def __init__(
        self,
        model,
        device,
        c_puct: float = DEFAULT_C_PUCT,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
        noise_weight: float = DEFAULT_NOISE_WEIGHT,
        batch_size: int = 32,
        candidate_radius: int = DEFAULT_CANDIDATE_RADIUS,
        candidate_width: int = DEFAULT_CANDIDATE_WIDTH,
        time_budget: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight
        self.batch_size = max(1, batch_size)
        self.candidate_radius = candidate_radius
        self.candidate_width = max(1, candidate_width)
        self.time_budget = time_budget
        self.rng = random.Random(seed)


    def search(
        self,
        board: Board,
        player: int,
        simulations: int = 400,
        add_noise: bool = False,
        last_move: tuple[int, int] | None = None,
        time_budget: float | None = None,
    ) -> SearchResult:
        root = Node(board.copy(), player)
        root.last_move = last_move


        self._expand(root, self._evaluate([root])[0])
        if add_noise:
            self._add_root_noise(root)

        pending: list[tuple[Node, list[Node]]] = []
        waiting: set[int] = set()

        budget = time_budget if time_budget is not None else self.time_budget
        deadline = time.monotonic() + budget if budget else None

        done = 0
        while done < simulations:
            leaf, path = self._descend(root, waiting)
            if leaf is None:
                self._release(path)
                self._flush(pending)
                waiting.clear()
                leaf, path = self._descend(root, waiting)
                if leaf is None:
                    break
            if leaf.terminal_value is not None:
                self._backpropagate(path, leaf.terminal_value)
                done += 1
                continue
            pending.append((leaf, path))
            waiting.add(id(leaf))
            if len(pending) >= self.batch_size:
                self._flush(pending)
                waiting.clear()
            done += 1
            if deadline is not None and done % 16 == 0 and time.monotonic() >= deadline:
                break

        if pending:
            self._flush(pending)

        return self._collect(root)


    def _evaluate(self, nodes: list[Node]) -> np.ndarray:
        planes = []
        masks = []
        for node in nodes:
            planes.append(encode(node.board, node.player, node.last_move))
            masks.append(legal_mask(node.board, self.candidate_radius))
        policies, _ = evaluate(self.model, np.stack(planes), np.stack(masks), self.device)
        return policies

    def _flush(self, pending: list[tuple[Node, list[Node]]]) -> None:
        if not pending:
            return
        planes = []
        masks = []
        for leaf, _ in pending:
            planes.append(encode(leaf.board, leaf.player, leaf.last_move))
            masks.append(legal_mask(leaf.board, self.candidate_radius))
        policies, values = evaluate(
            self.model, np.stack(planes), np.stack(masks), self.device
        )

        for (leaf, path), policy, value in zip(pending, policies, values):
            self._expand(leaf, policy)
            self._backpropagate(path, value)
        pending.clear()

    def _expand(self, node: Node, policy: list[float]) -> None:
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
        scored = sorted(
            zip(moves, priors),
            key=lambda item: item[1],
            reverse=True,
        )
        keep = scored[: self.candidate_width]
        for (row, col), prior in keep:
            won = node.board.wins_with(row, col, node.player)
            child = Node(None, other(node.player), parent=node, move=(row, col))
            child.prior = prior
            child.last_move = (row, col)
            if won:
                child.terminal_value = -1.0
            node.children[cell_of(row, col)] = child
        if node.board.is_full():
            node.terminal_value = 0.0

    def _add_root_noise(self, root: Node) -> None:
        moves = list(root.children)
        if not moves:
            return
        priors = [root.children[move].prior for move in moves]
        noisy = apply_noise(priors, self.dirichlet_alpha, self.noise_weight, self.rng)
        for move, prior in zip(moves, noisy):
            root.children[move].prior = prior


    def _descend(self, root: Node, waiting: set[int]) -> tuple[Node | None, list[Node]]:
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
        sqrt_total = math.sqrt(max(node.visits, 1))
        best_move: int | None = None
        best_score = -math.inf
        for move, child in node.children.items():
            if id(child) in waiting:
                continue
            if child.visits > 0:
                quality = -child.value_sum / child.visits
            else:
                quality = 0.0


            exploration = (
                self.c_puct * child.prior * sqrt_total / (1 + child.visits + child.virtual_visits)
            )
            score = quality + exploration
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


    def _release(self, path: list[Node]) -> None:
        for node in path:
            node.virtual_visits = max(0, node.virtual_visits - 1)

    def _backpropagate(self, path: list[Node], value: float) -> None:
        current = value
        for depth in range(len(path) - 1, -1, -1):
            node = path[depth]
            node.virtual_visits = max(0, node.virtual_visits - 1)
            node.visits += 1
            node.value_sum += current
            current = -current

    def _collect(self, root: Node) -> SearchResult:
        visits: dict[int, int] = {}
        policy = np.zeros(ACTION_COUNT, dtype=np.float32)
        total = 0
        for move, child in root.children.items():
            visits[move] = child.visits
            total += child.visits
        if total > 0:
            for move, child in root.children.items():
                policy[move] = child.visits / total


        root_value = root.value_sum / root.visits if root.visits else 0.0
        return SearchResult(visits, policy, root_value, total)
