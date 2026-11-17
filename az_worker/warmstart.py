from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .dataset import TensorDataset
from .network import GomokuNet, evaluate, load_model, save, train_tensors
from .selfplay import Example
from .teacher import (
    VALUE_EVAL,
    VALUE_OUTCOME,
    VALUE_SEARCH,
    collect_az_game,
    collect_one_game,
)
from .train import resolve_device


DEFAULT_GAMES = 100
DEFAULT_DEPTH = 4
DEFAULT_WIDTH = 12
DEFAULT_TIME = 0.05
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
DEFAULT_MODEL = "models/az/warmstart.pt"
DEFAULT_DATA_OUT = "models/az/teacher_data.pt"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


DEFAULT_TEACHER = "classical"


DEFAULT_TEACHER_MODEL = "models/az/current.pt"


DEFAULT_TEACHER_SIMULATIONS = 400


DEFAULT_TEACHER_TIME = 1.0


DEFAULT_TEACHER_WIDTH = 16


DEFAULT_TEACHER_MOVES = 8


DEFAULT_DECAY = 2.0


DEFAULT_SCORE_SCALE = 8_000.0


DEFAULT_VALUE_TARGET = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用传统引擎或旧 AZ 模型当老师，预热神经网络")
    parser.add_argument(
        "--teacher",
        choices=("classical", "az"),
        default=DEFAULT_TEACHER,
        help="老师是谁：classical 是传统引擎，az 是旧的自对弈权重（更强的搜索当老师）",
    )
    parser.add_argument(
        "--teacher-model",
        default=DEFAULT_TEACHER_MODEL,
        help="--teacher az 时的老师权重，通常是上一代的 champion",
    )
    parser.add_argument(
        "--teacher-simulations",
        type=int,
        default=DEFAULT_TEACHER_SIMULATIONS,
        help="--teacher az 时每手 MCTS 模拟数",
    )
    parser.add_argument(
        "--teacher-time",
        type=float,
        default=DEFAULT_TEACHER_TIME,
        help="--teacher az 时每手秒数上限",
    )
    parser.add_argument(
        "--teacher-width",
        type=int,
        default=DEFAULT_TEACHER_WIDTH,
        help="--teacher az 时每个节点保留的候选数",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="棋局数")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="传统引擎的搜索深度")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="传统引擎的宽度")
    parser.add_argument("--time", type=float, default=DEFAULT_TIME, help="传统引擎每手秒数")
    parser.add_argument("--teacher-moves", type=int, default=DEFAULT_TEACHER_MOVES, help="老师排名的着法数")
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY, help="着法权重的衰减")
    parser.add_argument("--score-scale", type=float, default=DEFAULT_SCORE_SCALE, help="分数到价值的换算")
    parser.add_argument(
        "--value-target",
        choices=(VALUE_OUTCOME, VALUE_EVAL, VALUE_SEARCH),
        default=DEFAULT_VALUE_TARGET,
        help="价值目标：outcome 是整局结果，eval 是传统引擎的评估分，"
        "search 是旧 AZ 模型的搜索值；不写则按老师取默认",
    )
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="网络通道数")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS, help="残差块数量")
    parser.add_argument("--steps", type=int, default=None, help="训练步数，默认按样本数定")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每步的样本数")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="并行收集老师样本的进程数，老师的搜索是纯 CPU 的",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="权重文件路径")
    parser.add_argument("--data-out", default=DEFAULT_DATA_OUT, help="把收集到的样本存下来，供自对弈训练混用")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser


def collect_examples(
    games: int,
    depth: int,
    width: int,
    time_budget: float,
    teacher_moves: int,
    decay: float,
    score_scale: float = DEFAULT_SCORE_SCALE,
    value_target: str = VALUE_OUTCOME,
    workers: int = 1,
) -> list[Example]:
    payload = (depth, width, time_budget, teacher_moves, decay, score_scale, value_target)
    if workers > 1 and games > 1:
        return _collect_parallel(games, payload, workers)
    examples: list[Example] = []
    for _ in range(games):
        examples.extend(collect_one_game(*payload))
    return examples


def _collect_parallel(
    games: int,
    payload: tuple,
    workers: int,
) -> list[Example]:
    collected: list[list[Example]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect_one_game, *payload) for _ in range(games)]
        for future in futures:
            collected.append(future.result())
    return [example for batch in collected for example in batch]


def resolve_value_target(teacher: str, requested: str | None) -> str:
    if requested is None:
        return VALUE_SEARCH if teacher == "az" else VALUE_EVAL
    if teacher == "az" and requested == VALUE_EVAL:
        raise ValueError("AZ 老师的价值目标只能是 search 或 outcome")
    if teacher == "classical" and requested == VALUE_SEARCH:
        raise ValueError("传统老师的价值目标只能是 eval 或 outcome")
    return requested


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed if args.seed is not None else 0)
    device = resolve_device(args.device)
    try:
        value_target = resolve_value_target(args.teacher, args.value_target)
    except ValueError as error:
        print(error, flush=True)
        return 2

    model_path = Path(args.model)
    if model_path.is_file():
        model = load_model(str(model_path), device)
        print(f"读取权重: {model_path}", flush=True)
    else:
        model = GomokuNet(channels=args.channels, blocks=args.blocks).to(device)
        print(f"没有权重文件，从随机网络开始: {model_path}", flush=True)

    if args.teacher == "az":
        if args.workers > 1:
            print("AZ 老师走后端 GPU 推理，单进程采集；忽略 --workers", flush=True)
        teacher_path = Path(args.teacher_model)
        if not teacher_path.is_file():
            print(f"找不到老师权重: {teacher_path}", flush=True)
            return 1
        teacher_model = load_model(str(teacher_path), device)
        print(
            f"老师: AZ {teacher_path}，{args.teacher_simulations} 次模拟"
            f" / {args.teacher_time} 秒 / 宽度 {args.teacher_width}，价值目标 {value_target}",
            flush=True,
        )
        examples: list[Example] = []
        for game in range(args.games):
            seed = None if args.seed is None else args.seed + game
            examples.extend(
                collect_az_game(
                    teacher_model,
                    device,
                    args.teacher_simulations,
                    args.teacher_time,
                    args.teacher_width,
                    value_target,
                    seed=seed,
                )
            )
    else:
        print(f"老师: 传统引擎 d{args.depth}/{args.time} 秒，价值目标 {value_target}", flush=True)
        examples = collect_examples(
            args.games,
            args.depth,
            args.width,
            args.time,
            args.teacher_moves,
            args.decay,
            args.score_scale,
            value_target,
            args.workers,
        )


    held_out = max(1, len(examples) // 10)
    validation = examples[-held_out:]
    examples = examples[:-held_out]
    print(f"样本: {len(examples)}（另留 {len(validation)} 个做验证）", flush=True)
    values = [example.value for example in examples]
    print(
        f"价值范围: {min(values):.3f} 到 {max(values):.3f}，"
        f"平均 {sum(values) / len(values):.3f}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = args.steps if args.steps is not None else max(1, len(examples) // 32) * 6
    dataset = TensorDataset(device, capacity=len(examples))
    dataset.extend(examples)
    print(f"样本已放到显卡上: {len(dataset)} 条", flush=True)

    batch = args.batch_size
    for step in range(steps):
        planes, policy, legal, targets = dataset.sample(batch, rng, augment=True)
        loss, policy_loss, value_loss = train_tensors(
            model,
            optimizer,
            planes,
            policy,
            targets,
            legal,
        )
        if (step + 1) % 50 == 0:
            agreement = agreement_of(model, validation, device)
            print(
                f"  第 {step + 1}/{steps} 步，损失 {loss:.4f}"
                f"（策略 {policy_loss:.4f}，价值 {value_loss:.4f}），"
                f"验证首选一致 {agreement:.0%}",
                flush=True,
            )

    agreement = agreement_of(model, validation, device)
    print(f"验证首选一致率: {agreement:.0%}", flush=True)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(model_path))
    print(f"预热权重已写入: {model_path}", flush=True)

    if args.data_out:
        dataset.extend(validation)
        dataset.save(args.data_out)
        print(f"样本已写入: {args.data_out}（{len(dataset)} 条）", flush=True)
    return 0


def agreement_of(model: GomokuNet, examples: list[Example], device: torch.device) -> float:
    if not examples:
        return 0.0
    matches = 0
    for start in range(0, len(examples), 128):
        batch = examples[start : start + 128]
        policies, _ = evaluate(
            model,
            np.stack([example.planes for example in batch]),
            np.stack([example.legal for example in batch]),
            device,
        )
        for policy, example in zip(policies, batch):
            best = int(np.argmax(policy))
            if example.teacher_cell >= 0 and best == example.teacher_cell:
                matches += 1
    return matches / len(examples)


if __name__ == "__main__":
    raise SystemExit(main())
