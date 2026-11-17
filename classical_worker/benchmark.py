from __future__ import annotations

import argparse
import statistics
import time

from .config import DEFAULT_DEPTH, DEFAULT_VCF_BUDGET, DEFAULT_WIDTH
from .position import BLACK, Position, WHITE, other
from .search import Searcher
from .tests.fixtures.v1_position import Position as V1Position
from .tests.fixtures.v1_search import Searcher as V1Searcher


OPENINGS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))

# 基准是回归测量，用固定的快档参数；生产默认值只有 depth/width 与 VCF 比例相同。
BENCHMARK_BUDGET = 0.2
BENCHMARK_VCF_BUDGET = DEFAULT_VCF_BUDGET
V1_DEPTH = 6
V1_WIDTH = 10


def play_game(
    v2_black: bool,
    budget: float,
    opening: tuple[int, int],
    vcf_budget: float = BENCHMARK_VCF_BUDGET,
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
        depth=DEFAULT_DEPTH,
        width=DEFAULT_WIDTH,
        time_budget=budget,
        vcf_budget=vcf_budget,
    )
    v1 = V1Searcher(depth=V1_DEPTH, width=V1_WIDTH, time_budget=budget)
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
    vcf_budget: float = BENCHMARK_VCF_BUDGET,
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
            v2_black, budget, opening, vcf_budget
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
    parser.add_argument("--budget", type=float, default=BENCHMARK_BUDGET)
    parser.add_argument("--vcf", type=float, default=BENCHMARK_VCF_BUDGET)
    args = parser.parse_args(argv)
    result = benchmark(
        max(1, args.games),
        max(0.01, args.budget),
        max(0.0, args.vcf),
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
