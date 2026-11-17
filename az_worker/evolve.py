from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import arena, train, warmstart


DEFAULT_CHAMPION = "models/az/current_v3.pt"
DEFAULT_OUT_DIR = "models/az/evolution"
DEFAULT_GENERATIONS = 10
DEFAULT_PROMOTE_WINRATE = 0.55
DEFAULT_MIN_BASELINE_WINRATE = 0.45
DEFAULT_MATCH_GAMES = 40
DEFAULT_MATCH_SIMULATIONS = 100_000
DEFAULT_MATCH_TIME = 2.0
DEFAULT_MATCH_WIDTH = 16
DEFAULT_BASELINE_GAMES = 20
DEFAULT_OPENING_MOVES = 2
DEFAULT_DISTILL_GAMES = 60
DEFAULT_DISTILL_SIMULATIONS = 400
DEFAULT_DISTILL_TIME = 1.0
DEFAULT_DISTILL_WIDTH = 16
DEFAULT_DISTILL_STEPS = 1000
DEFAULT_DISTILL_BATCH = 1024
DEFAULT_DISTILL_LR = 1e-4
DEFAULT_GAMES = 10
DEFAULT_ITERATIONS = 20
DEFAULT_SIMULATIONS = 400
DEFAULT_TRAIN_STEPS = 300
DEFAULT_BATCH = 1024
DEFAULT_BUFFER = 60_000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DEVICE = "cuda"


def should_promote(result: arena.MatchResult, threshold: float) -> bool:
    if not result.games:
        return False
    return result.winrate >= threshold


def decide_promotion(
    gate: arena.MatchResult,
    threshold: float,
    baseline: arena.MatchResult | None = None,
    min_baseline_winrate: float = 0.0,
) -> bool:
    if not should_promote(gate, threshold):
        return False
    if baseline is None or not baseline.games:
        return True
    return baseline.winrate >= min_baseline_winrate


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def advance_state(
    state: dict,
    generation: int,
    teacher: Path,
    candidate: Path,
    promoted: bool,
    gate: arena.MatchResult,
    baseline: arena.MatchResult | None,
) -> None:
    if promoted:
        state["champion"] = str(candidate)
    state["learner"] = str(candidate)
    state["generation"] = generation
    entry = {
        "generation": generation,
        "teacher": str(teacher),
        "candidate": str(candidate),
        "wins": gate.wins,
        "losses": gate.losses,
        "draws": gate.draws,
        "winrate": round(gate.winrate, 4),
        "promoted": promoted,
    }
    if baseline is not None:
        entry["baseline_wins"] = baseline.wins
        entry["baseline_losses"] = baseline.losses
        entry["baseline_draws"] = baseline.draws
        entry["baseline_winrate"] = round(baseline.winrate, 4)
    state["history"].append(entry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自进化：champion 当老师和数据生成器，学习器持续累积，擂台达标才晋级",
    )
    parser.add_argument("--champion", default=DEFAULT_CHAMPION, help="初始 champion 权重")
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="固定基准权重，每代都让候选与它打一次，用来观察是否真的进步；默认是初始 champion",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="产物的目录与 state.json")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="跑多少代")
    parser.add_argument("--promote-winrate", type=float, default=DEFAULT_PROMOTE_WINRATE, help="晋级胜率门槛（和棋算半胜）")
    parser.add_argument(
        "--min-baseline-winrate",
        type=float,
        default=DEFAULT_MIN_BASELINE_WINRATE,
        help="对固定基准的胜率下限；赢了 champion 但低于它就不晋级，防止代际互克导致的退化",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，按代递增")
    parser.add_argument("--resume", action="store_true", help="从 out-dir/state.json 继续")
    parser.add_argument("--match-games", type=int, default=DEFAULT_MATCH_GAMES, help="擂台局数")
    parser.add_argument(
        "--match-simulations",
        type=int,
        default=DEFAULT_MATCH_SIMULATIONS,
        help="擂台每手模拟次数",
    )
    parser.add_argument("--match-time", type=float, default=DEFAULT_MATCH_TIME, help="擂台每手秒数上限")
    parser.add_argument("--match-width", type=int, default=DEFAULT_MATCH_WIDTH, help="擂台每个节点保留的候选数")
    parser.add_argument(
        "--baseline-games",
        type=int,
        default=DEFAULT_BASELINE_GAMES,
        help="每代候选对固定基准的局数，0 表示不做对照组",
    )
    parser.add_argument(
        "--opening-moves",
        type=int,
        default=DEFAULT_OPENING_MOVES,
        help="擂台每对棋局开头的随机着法数，避免两个确定性引擎重复同一局",
    )
    parser.add_argument("--distill-games", type=int, default=DEFAULT_DISTILL_GAMES, help="champion 当老师采集的棋局数")
    parser.add_argument(
        "--distill-simulations",
        type=int,
        default=DEFAULT_DISTILL_SIMULATIONS,
        help="老师每手模拟次数",
    )
    parser.add_argument("--distill-time", type=float, default=DEFAULT_DISTILL_TIME, help="老师每手秒数上限")
    parser.add_argument("--distill-width", type=int, default=DEFAULT_DISTILL_WIDTH, help="老师每个节点保留的候选数")
    parser.add_argument("--distill-steps", type=int, default=DEFAULT_DISTILL_STEPS, help="蒸馏训练步数")
    parser.add_argument("--distill-batch", type=int, default=DEFAULT_DISTILL_BATCH, help="蒸馏批大小")
    parser.add_argument("--distill-lr", type=float, default=DEFAULT_DISTILL_LR, help="蒸馏学习率")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="每轮自对弈局数（由 champion 生成）")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="自对弈训练轮数")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="自对弈生成器每手模拟次数")
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS, help="每轮训练步数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="训练批大小")
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER, help="回放缓冲区容量，跨代保存")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="学习率")
    return parser


def play_match(
    first: Path,
    second: Path,
    games: int,
    args: argparse.Namespace,
    seed: int | None,
) -> arena.MatchResult:
    candidate = arena.az_side(
        first,
        args.device,
        args.match_simulations,
        args.match_time,
        args.match_width,
        seed=seed,
    )
    opponent = arena.az_side(
        second,
        args.device,
        args.match_simulations,
        args.match_time,
        args.match_width,
        seed=None if seed is None else seed + 1,
    )
    return arena.play_series(candidate, opponent, games, args.opening_moves, seed)


def run_generation(args: argparse.Namespace, out_dir: Path, state: dict) -> bool:
    generation = int(state["generation"]) + 1
    champion = Path(state["champion"])
    learner = Path(state.get("learner") or state["champion"])
    if not learner.is_file():
        learner = champion
    baseline = Path(state.get("baseline") or state["champion"])

    warm_path = out_dir / f"gen_{generation:03d}_warm.pt"
    teacher_data = out_dir / f"gen_{generation:03d}_teacher.pt"
    candidate_path = out_dir / f"gen_{generation:03d}_candidate.pt"
    for stale in (warm_path, teacher_data, candidate_path):
        stale.unlink(missing_ok=True)
    shutil.copyfile(learner, warm_path)
    seed = None if args.seed is None else args.seed + generation

    warm_args = [
        "--teacher",
        "az",
        "--teacher-model",
        str(champion),
        "--games",
        str(args.distill_games),
        "--teacher-simulations",
        str(args.distill_simulations),
        "--teacher-time",
        str(args.distill_time),
        "--teacher-width",
        str(args.distill_width),
        "--steps",
        str(args.distill_steps),
        "--lr",
        str(args.distill_lr),
        "--batch-size",
        str(args.distill_batch),
        "--device",
        args.device,
        "--model",
        str(warm_path),
        "--data-out",
        str(teacher_data),
    ]
    if seed is not None:
        warm_args += ["--seed", str(seed)]
    print(
        f"=== 第 {generation} 代：学习器 {learner.name} 从 {champion.name} 蒸馏 ===",
        flush=True,
    )
    if warmstart.main(warm_args):
        print("蒸馏失败", flush=True)
        return False

    train_args = [
        "--init",
        str(warm_path),
        "--model",
        str(candidate_path),
        "--selfplay-model",
        str(champion),
        "--teacher-data",
        str(teacher_data),
        "--games",
        str(args.games),
        "--iterations",
        str(args.iterations),
        "--simulations",
        str(args.simulations),
        "--train-steps",
        str(args.train_steps),
        "--batch-size",
        str(args.batch_size),
        "--buffer",
        str(args.buffer),
        "--buffer-file",
        str(out_dir / "replay.pt"),
        "--resume",
        "--lr",
        str(args.lr),
        "--device",
        args.device,
    ]
    if seed is not None:
        train_args += ["--seed", str(seed)]
    print(f"=== 第 {generation} 代：自对弈训练 {candidate_path.name} ===", flush=True)
    if train.main(train_args):
        print("自对弈训练失败", flush=True)
        return False

    match_seed = None if seed is None else seed + 1
    print(f"=== 第 {generation} 代：{candidate_path.name} 对 {champion.name} ===", flush=True)
    gate = play_match(candidate_path, champion, args.match_games, args, match_seed)
    print(
        f"第 {generation} 代: {gate.wins} 胜 {gate.losses} 负 {gate.draws} 和，"
        f"胜率 {gate.winrate:.0%}，门槛 {args.promote_winrate:.0%}",
        flush=True,
    )

    benchmark = None
    if args.baseline_games > 0:
        print(f"=== 第 {generation} 代：{candidate_path.name} 对固定基准 {baseline.name} ===", flush=True)
        benchmark = play_match(
            candidate_path,
            baseline,
            args.baseline_games,
            args,
            None if match_seed is None else match_seed + 2,
        )
        print(
            f"第 {generation} 代对照: {benchmark.wins} 胜 {benchmark.losses} 负 "
            f"{benchmark.draws} 和，对基准胜率 {benchmark.winrate:.0%}",
            flush=True,
        )

    promoted = decide_promotion(
        gate,
        args.promote_winrate,
        benchmark,
        args.min_baseline_winrate,
    )
    if promoted:
        verdict = "晋级"
    elif not should_promote(gate, args.promote_winrate):
        verdict = "淘汰（没赢 champion）"
    else:
        verdict = f"淘汰（对基准低于 {args.min_baseline_winrate:.0%}）"
    print(f"第 {generation} 代判定: {verdict}", flush=True)

    advance_state(state, generation, champion, candidate_path, promoted, gate, benchmark)
    save_state(out_dir / "state.json", state)
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.promote_winrate <= 1.0:
        print("--promote-winrate 必须在 0 到 1 之间", flush=True)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    if state_path.is_file():
        if not args.resume:
            print(f"{state_path} 已存在；加 --resume 继续，或换 --out-dir", flush=True)
            return 2
        state = load_state(state_path)
        print(
            f"继续进化：已完成 {state['generation']} 代，champion 是 {state['champion']}，"
            f"学习器是 {state.get('learner', state['champion'])}",
            flush=True,
        )
    else:
        champion = Path(args.champion)
        if not champion.is_file():
            print(f"找不到初始 champion: {champion}", flush=True)
            return 1
        state = {
            "generation": 0,
            "champion": str(champion),
            "learner": str(champion),
            "baseline": str(Path(args.baseline_model) if args.baseline_model else champion),
            "history": [],
        }
        save_state(state_path, state)
        print(f"新进化：champion 与学习器都是 {champion}，基准是 {state['baseline']}", flush=True)

    if args.baseline_model:
        state["baseline"] = str(Path(args.baseline_model))

    for name in ("champion", "learner", "baseline"):
        path = Path(state.get(name) or state["champion"])
        if not path.is_file():
            print(f"找不到 {name}: {path}", flush=True)
            return 1
    if args.baseline_games < 0:
        print("--baseline-games 不能为负", flush=True)
        return 2

    for _ in range(args.generations):
        if not run_generation(args, out_dir, state):
            return 1
    print(f"完成 {args.generations} 代，champion 是 {state['champion']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
