from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from .dataset import TensorDataset
from .network import GomokuNet, load_model, read_meta, train_batch
from .selfplay import TEMPERATURE_MOVES, SelfPlay, save_checkpoint


TOTAL_STEPS = 25_000

ITERATIONS = 100

GAMES_PER_ITERATION = 10

BATCH_SIZE = 256

LEARNING_RATE = 0.0125

MOMENTUM = 0.9

WEIGHT_DECAY = 1e-4

BUFFER_CAPACITY = 100_000

CHANNELS = 64

BLOCKS = 6

VALUE_HIDDEN = 64

SEARCH_BATCH = 32

SIMULATIONS_SCHEDULE = (128, 256, 400)

SIMULATIONS_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)

DECAY_FRACTIONS = (1.0 / 7.0, 3.0 / 7.0, 5.0 / 7.0)

DEFAULT_MODEL_PATH = "models/az/current.pt"


def learning_rate(step: int, total_steps: int, base: float = LEARNING_RATE) -> float:
    decays = 0
    for fraction in DECAY_FRACTIONS:
        if step >= math.floor(total_steps * fraction):
            decays += 1
    return base * (0.1**decays)


def simulations_for(step: int, total_steps: int) -> int:
    share = step / max(1, total_steps)
    if share < SIMULATIONS_FRACTIONS[0]:
        return SIMULATIONS_SCHEDULE[0]
    if share < SIMULATIONS_FRACTIONS[1]:
        return SIMULATIONS_SCHEDULE[1]
    return SIMULATIONS_SCHEDULE[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero 纯自对弈训练")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS, help="总梯度步数")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="训练轮数")
    parser.add_argument("--games", type=int, default=GAMES_PER_ITERATION, help="每轮自对弈局数")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="初始学习率")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="SGD 动量")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 系数")
    parser.add_argument("--buffer", type=int, default=BUFFER_CAPACITY, help="回放缓冲区容量")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="固定每手模拟数；默认按步数进度用 128/256/400",
    )
    parser.add_argument("--search-batch", type=int, default=SEARCH_BATCH, help="MCTS 批大小")
    parser.add_argument(
        "--temperature-moves", type=int, default=TEMPERATURE_MOVES, help="前多少手用温度采样"
    )
    parser.add_argument("--augment", action="store_true", help="开启八对称数据增强（默认关闭）")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="权重输出路径")
    parser.add_argument(
        "--init", default=None, help="起始权重，默认与 --model 相同；分开写可续训"
    )
    parser.add_argument("--channels", type=int, default=CHANNELS, help="残差塔通道数")
    parser.add_argument("--blocks", type=int, default=BLOCKS, help="残差块数量")
    parser.add_argument("--value-hidden", type=int, default=VALUE_HIDDEN, help="价值头隐层宽度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--buffer-file", default=None, help="回放缓冲区存档文件")
    parser.add_argument("--resume", action="store_true", help="读取缓冲区与步数继续训练")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_steps < 1 or args.iterations < 1:
        print("--total-steps 与 --iterations 必须为正", flush=True)
        return 2
    device = resolve_device(args.device)
    seed = args.seed if args.seed is not None else 0
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model_path = Path(args.model)
    start_path = Path(args.init) if args.init else model_path
    if start_path.is_file():
        model = load_model(str(start_path), device)
    else:
        model = GomokuNet(
            channels=args.channels, blocks=args.blocks, value_hidden=args.value_hidden
        ).to(device)
        print(f"没有权重文件，从随机网络开始: {start_path}", flush=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    step = 0
    dataset = TensorDataset(device, capacity=args.buffer)
    if args.resume and args.buffer_file:
        dataset = TensorDataset.load(args.buffer_file, device, args.buffer)
        if model_path.is_file():
            step = int(read_meta(str(model_path)).get("step", 0))
        print(f"续训: 缓冲区 {len(dataset)} 条，已完成 {step} 步", flush=True)

    steps_per_iteration = max(1, math.ceil(args.total_steps / args.iterations))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"设备 {device}，参数 {parameters}，总步数 {args.total_steps}，"
        f"每轮 {steps_per_iteration} 步，轮数 {args.iterations}",
        flush=True,
    )

    for iteration in range(1, args.iterations + 1):
        if step >= args.total_steps:
            break
        started = time.monotonic()
        simulations = args.simulations or simulations_for(step, args.total_steps)
        player = SelfPlay(
            model,
            device,
            simulations=simulations,
            batch_size=args.search_batch,
            temperature_moves=args.temperature_moves,
            seed=rng.randrange(1 << 30),
        )
        wins = {1: 0, 2: 0, None: 0}
        plies = 0
        for _ in range(args.games):
            game = player.play_game()
            dataset.extend(game.examples)
            wins[game.winner] = wins.get(game.winner, 0) + 1
            plies += game.moves

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        for _ in range(steps_per_iteration):
            if step >= args.total_steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.total_steps, args.lr)
            planes, target_policy, mask, target_value = dataset.sample(
                args.batch_size, rng, augment=args.augment
            )
            loss, policy_loss, value_loss = train_batch(
                model,
                optimizer,
                planes,
                target_policy,
                target_value,
                mask,
            )
            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            step += 1

        save_checkpoint(
            model,
            model_path,
            {
                "step": step,
                "total_steps": args.total_steps,
                "lr": learning_rate(step, args.total_steps, args.lr),
                "simulations": simulations,
            },
        )
        if args.buffer_file:
            dataset.save(args.buffer_file)
        elapsed = time.monotonic() - started
        recent = len(policy_losses[-20:]) or 1
        print(
            f"第 {iteration} 轮: 步 {step}/{args.total_steps}，学习率 "
            f"{learning_rate(step, args.total_steps, args.lr):.6g}，模拟 {simulations}，"
            f"自对弈 {args.games} 局（{wins[1]}:{wins[2]}，和 {wins[None]}，"
            f"{plies} 手），缓冲区 {len(dataset)}，"
            f"策略 {sum(policy_losses[-20:]) / recent:.4f} "
            f"价值 {sum(value_losses[-20:]) / recent:.4f}，"
            f"用时 {elapsed:.0f} 秒",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
