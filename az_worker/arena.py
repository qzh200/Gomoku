from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from classical_worker.position import Position
from classical_worker.search import Searcher

from .board import BLACK, WHITE, Board, other
from .engine import AlphaZeroEngine
from .mcts import DEFAULT_CANDIDATE_WIDTH


DEFAULT_GAMES = 10
DEFAULT_SIMULATIONS = 100_000
DEFAULT_OPPONENT = "classical"
DEFAULT_OPPONENT_MODEL = "models/az/current.pt"
DEFAULT_CLASSICAL_DEPTH = 4
DEFAULT_CLASSICAL_WIDTH = 12
DEFAULT_CLASSICAL_TIME = 0.5
DEFAULT_OPENING_MOVES = 2


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def winrate(self) -> float:
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


class AzSide:
    def __init__(self, engine: AlphaZeroEngine) -> None:
        self.engine = engine

    def choose(
        self,
        board: Board,
        player: int,
        last_move: dict | None,
    ) -> tuple[int, int]:
        return self.engine.choose_move(
            [row[:] for row in board.grid],
            [],
            player,
            last_move,
            {},
        )


class ClassicalSide:
    def __init__(self, searcher: Searcher) -> None:
        self.searcher = searcher

    def choose(
        self,
        board: Board,
        player: int,
        last_move: dict | None,
    ) -> tuple[int, int] | None:
        return self.searcher.best_move(Position([row[:] for row in board.grid]), player)


def az_side(
    model_path: str | Path,
    device: str,
    simulations: int,
    time_budget: float | None,
    candidate_width: int,
    seed: int | None = None,
) -> AzSide:
    return AzSide(
        AlphaZeroEngine(
            model_path=model_path,
            device=device,
            simulations=simulations,
            candidate_width=candidate_width,
            time_budget=time_budget,
            seed=seed,
        )
    )


def classical_side(depth: int, width: int, time_budget: float) -> ClassicalSide:
    return ClassicalSide(Searcher(depth=depth, width=width, time_budget=time_budget))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero 擂台赛")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="棋局数")
    parser.add_argument("--model", default="models/az/current.pt", help="候选模型权重文件")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="候选每手模拟次数")
    parser.add_argument(
        "--candidate-width",
        type=int,
        default=DEFAULT_CANDIDATE_WIDTH,
        help="候选每个节点保留的候选数",
    )
    parser.add_argument("--time-budget", type=float, default=None, help="候选每手秒数上限")
    parser.add_argument(
        "--opponent",
        choices=("classical", "az"),
        default=DEFAULT_OPPONENT,
        help="对手是谁：classical 是传统引擎，az 是旧的 AZ 权重",
    )
    parser.add_argument(
        "--opponent-model",
        default=DEFAULT_OPPONENT_MODEL,
        help="--opponent az 时的对手权重，通常是上一代的 champion",
    )
    parser.add_argument(
        "--opponent-simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help="--opponent az 时对手每手模拟次数",
    )
    parser.add_argument(
        "--opponent-width",
        type=int,
        default=DEFAULT_CANDIDATE_WIDTH,
        help="--opponent az 时对手每个节点保留的候选数",
    )
    parser.add_argument(
        "--opponent-time",
        type=float,
        default=None,
        help="--opponent az 时对手每手秒数上限，默认与 --time-budget 相同",
    )
    parser.add_argument("--classical-depth", type=int, default=DEFAULT_CLASSICAL_DEPTH)
    parser.add_argument("--classical-width", type=int, default=DEFAULT_CLASSICAL_WIDTH)
    parser.add_argument("--classical-time", type=float, default=DEFAULT_CLASSICAL_TIME)
    parser.add_argument(
        "--opening-moves",
        type=int,
        default=DEFAULT_OPENING_MOVES,
        help="每对棋局开头的随机着法数；两个确定性引擎重复同一局会毁掉统计，0 表示关闭",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser


def random_opening(rng: random.Random, plies: int) -> list[tuple[int, int]]:
    board = Board()
    player = BLACK
    opening: list[tuple[int, int]] = []
    for _ in range(max(0, plies)):
        cells = board.candidates(2)
        if not cells:
            break
        row, col = cells[rng.randrange(len(cells))]
        won = board.play(row, col, player)
        opening.append((row, col))
        if won:
            break
        player = other(player)
    return opening


def play_game(
    candidate: AzSide,
    opponent: AzSide | ClassicalSide,
    candidate_is_black: bool,
    opening: list[tuple[int, int]] | None = None,
) -> int | None:
    board = Board()
    player = BLACK
    candidate_player = BLACK if candidate_is_black else WHITE
    last_move: dict | None = None
    for row, col in opening or ():
        last_move = {"row": row, "col": col}
        if board.play(row, col, player):
            return player
        player = other(player)
    while not board.is_full():
        side = candidate if player == candidate_player else opponent
        move = side.choose(board, player, last_move)
        if move is None:
            break
        row, col = move
        last_move = {"row": row, "col": col}
        if board.play(row, col, player):
            return player
        player = other(player)
    return None


def play_series(
    candidate: AzSide,
    opponent: AzSide | ClassicalSide,
    games: int,
    opening_moves: int = 0,
    seed: int | None = None,
    report=None,
) -> MatchResult:
    rng = random.Random(seed)
    wins = 0
    losses = 0
    draws = 0
    opening: list[tuple[int, int]] = []
    for game in range(games):
        if game % 2 == 0:
            opening = random_opening(rng, opening_moves)
        candidate_is_black = game % 2 == 0
        winner = play_game(candidate, opponent, candidate_is_black, opening)
        if winner is None:
            draws += 1
        elif winner == (BLACK if candidate_is_black else WHITE):
            wins += 1
        else:
            losses += 1
        if report is not None:
            report(game, winner, candidate_is_black)
    return MatchResult(wins, losses, draws)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    candidate = az_side(
        args.model,
        args.device,
        args.simulations,
        args.time_budget,
        args.candidate_width,
        seed=args.seed,
    )
    if args.opponent == "az":
        opponent_path = Path(args.opponent_model)
        if not opponent_path.is_file():
            print(f"找不到对手权重: {opponent_path}", flush=True)
            return 1
        opponent_time = args.opponent_time if args.opponent_time is not None else args.time_budget
        opponent = az_side(
            opponent_path,
            args.device,
            args.opponent_simulations,
            opponent_time,
            args.opponent_width,
            seed=None if args.seed is None else args.seed + 1,
        )
        opponent_name = f"AZ {opponent_path}"
    else:
        opponent = classical_side(
            args.classical_depth,
            args.classical_width,
            args.classical_time,
        )
        opponent_name = (
            f"传统引擎 d{args.classical_depth}/{args.classical_time} 秒"
        )

    def report(game: int, winner: int | None, candidate_is_black: bool) -> None:
        if winner is None:
            label = "和局"
        elif winner == (BLACK if candidate_is_black else WHITE):
            label = "胜"
        else:
            label = "负"
        print(f"第 {game + 1} 局: {label}", flush=True)

    result = play_series(
        candidate,
        opponent,
        args.games,
        args.opening_moves,
        args.seed,
        report,
    )
    elapsed = time.monotonic() - started
    print(
        f"候选对 {opponent_name}: {result.wins} 胜 {result.losses} 负 {result.draws} 和，"
        f"共 {result.games} 局，胜率 {result.winrate:.0%}，"
        f"开局随机着法 {args.opening_moves} 手，用时 {elapsed:.0f} 秒",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
