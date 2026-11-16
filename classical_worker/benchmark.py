from __future__ import annotations

import argparse
import statistics
import time

from .position import BLACK, Position, WHITE, other
from .search import Searcher
from .tests.fixtures.v1_position import Position as V1Position
from .tests.fixtures.v1_search import Searcher as V1Searcher


OPENINGS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def play_game(
    v2_black: bool,
    budget: float,
    opening: tuple[int, int],
    vcf_budget: float = 0.10,
    vct_budget: float = 0.0,
) -> tuple[int, int, list[float], list[int]]:
    position = Position()
    centre = (7, 7)
    first = position.play(*centre, BLACK)
    if first.win:
        raise AssertionError("the centre stone cannot win")
    second = position.play(centre[0] + opening[0], centre[1] + opening[1], WHITE)
    if second.win:
        raise AssertionError("the opening stone cannot win")

    v2 = Searcher(
        depth=8,
        width=12,
        time_budget=budget,
        vcf_budget=vcf_budget,
        vct_budget=vct_budget,
    )
    v1 = V1Searcher(depth=6, width=10, time_budget=budget)
    player = BLACK
    moves = 2
    times: list[float] = []
    depths: list[int] = []
    while not position.is_full():
        if (player == BLACK) == v2_black:
            started = time.monotonic()
            move = v2.best_move(position, player)
            times.append(time.monotonic() - started)
            depths.append(v2.depth_reached)
        else:
            move = v1.best_move(V1Position(position.grid), player)
        if move is None:
            break
        undo = position.play(*move, player)
        moves += 1
        if undo.win:
            return player, moves, times, depths
        player = other(player)
    return 0, moves, times, depths


def benchmark(
    games: int,
    budget: float,
    vcf_budget: float = 0.10,
    vct_budget: float = 0.0,
) -> dict[str, float]:
    wins = 0
    losses = 0
    draws = 0
    times: list[float] = []
    depths: list[int] = []
    started = time.monotonic()
    for game in range(games):
        v2_black = game % 2 == 0
        opening = OPENINGS[game % len(OPENINGS)]
        winner, _, game_times, game_depths = play_game(
            v2_black, budget, opening, vcf_budget, vct_budget
        )
        times.extend(game_times)
        depths.extend(game_depths)
        v2_player = BLACK if v2_black else WHITE
        if winner == 0:
            draws += 1
        elif winner == v2_player:
            wins += 1
        else:
            losses += 1
    elapsed = time.monotonic() - started
    times.sort()
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))] if times else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "average": statistics.fmean(times) if times else 0.0,
        "p95": p95,
        "maximum": times[-1] if times else 0.0,
        "depth": statistics.fmean(depths) if depths else 0.0,
        "elapsed": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第二代传统算法对第一代的自对弈基准")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--vcf", type=float, default=0.15)
    parser.add_argument("--vct", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = benchmark(
        max(1, args.games),
        max(0.01, args.budget),
        max(0.0, args.vcf),
        max(0.0, args.vct),
    )
    print(f"games      {result['games']}")
    print(f"wins       {result['wins']}")
    print(f"losses     {result['losses']}")
    print(f"draws      {result['draws']}")
    print(f"win rate   {result['win_rate']:.1%}")
    print(f"average    {result['average']:.3f} s")
    print(f"p95        {result['p95']:.3f} s")
    print(f"maximum    {result['maximum']:.3f} s")
    print(f"depth      {result['depth']:.1f}")
    print(f"elapsed    {result['elapsed']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
