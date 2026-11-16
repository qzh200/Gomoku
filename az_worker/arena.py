from __future__ import annotations

import argparse
import time

from classical_worker.position import Position
from classical_worker.search import Searcher

from .board import BLACK, WHITE, Board, other
from .engine import AlphaZeroEngine


DEFAULT_GAMES = 10
DEFAULT_SIMULATIONS = 400
DEFAULT_CLASSICAL_DEPTH = 4
DEFAULT_CLASSICAL_WIDTH = 12
DEFAULT_CLASSICAL_TIME = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero 与传统引擎的擂台赛")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="棋局数")
    parser.add_argument("--model", default="models/az/current.pt", help="AlphaZero 权重文件")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="每手模拟次数")
    parser.add_argument("--classical-depth", type=int, default=DEFAULT_CLASSICAL_DEPTH)
    parser.add_argument("--classical-width", type=int, default=DEFAULT_CLASSICAL_WIDTH)
    parser.add_argument("--classical-time", type=float, default=DEFAULT_CLASSICAL_TIME)
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser


def play_game(az: AlphaZeroEngine, classical: Searcher, az_is_black: bool) -> int | None:
    board = Board()
    player = BLACK
    az_player = BLACK if az_is_black else WHITE
    last_move: dict | None = None
    while not board.is_full():
        if player == az_player:
            row, col = az.choose_move(
                [line[:] for line in board.grid],
                [],
                player,
                last_move,
                {},
            )
        else:
            move = classical.best_move(Position([line[:] for line in board.grid]), player)
            if move is None:
                break
            row, col = move
        last_move = {"row": row, "col": col}
        if board.play(row, col, player):
            return player
        player = other(player)
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    az = AlphaZeroEngine(
        model_path=args.model,
        device=args.device,
        simulations=args.simulations,
        seed=args.seed,
    )
    classical = Searcher(
        depth=args.classical_depth,
        width=args.classical_width,
        time_budget=args.classical_time,
    )

    wins = 0
    losses = 0
    draws = 0
    started = time.monotonic()
    for game in range(args.games):
        az_is_black = game % 2 == 0
        winner = play_game(az, classical, az_is_black)
        if winner is None:
            draws += 1
            label = "和局"
        elif winner == (BLACK if az_is_black else WHITE):
            wins += 1
            label = "胜"
        else:
            losses += 1
            label = "负"
        print(f"第 {game + 1} 局: {label}", flush=True)
    elapsed = time.monotonic() - started
    print(
        f"AlphaZero 对传统引擎: {wins} 胜 {losses} 负 {draws} 和，"
        f"共 {args.games} 局，用时 {elapsed:.0f} 秒",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
