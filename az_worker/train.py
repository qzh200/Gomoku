"""The training of the AlphaZero network.

Usage:
    python -m az_worker.train --games 20 --iterations 10
    python -m az_worker.train --device cpu --simulations 64

Every iteration plays a few games with the current network, puts the examples
into the replay buffer and trains on them. The weights are written after
every iteration, so a run that is stopped keeps what it learned.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from .network import GomokuNet, load_model
from .selfplay import Example, ReplayBuffer, SelfPlay, evaluate, save_checkpoint, train

# The defaults of a training run.
DEFAULT_GAMES = 10
DEFAULT_ITERATIONS = 10
DEFAULT_SIMULATIONS = 200
DEFAULT_TRAIN_STEPS = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_BUFFER = 60_000
DEFAULT_MODEL_PATH = "models/az/current.pt"
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
DEFAULT_LEARNING_RATE = 1e-3


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
    parser = argparse.ArgumentParser(description="AlphaZero 自对弈训练")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="每轮自对弈的局数")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="训练轮数")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="每手搜索的模拟数")
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS, help="每轮训练步数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="训练批大小")
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER, help="回放缓冲区容量")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="网络通道数")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS, help="残差块数量")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="学习率")
    parser.add_argument("--temperature-moves", type=int, default=12, help="前多少手用温度采样")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT 常数")
    parser.add_argument("--candidate-radius", type=int, default=1, help="候选点半径")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="权重文件路径")
    parser.add_argument(
        "--init",
        default=None,
        help="起始权重文件，默认与 --model 相同；分开写就不会覆盖预热权重",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no-augment", action="store_true", help="不做对称增强")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有的 --model 继续，保留回放缓冲区（需要 --buffer-file）",
    )
    parser.add_argument("--buffer-file", default=None, help="回放缓冲区的存档文件")
    return parser


def resolve_device(name: str) -> torch.device:
    """Return the device the training runs on."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    """Train the network and write the weights after every iteration."""
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed if args.seed is not None else 0)

    model_path = Path(args.model)
    # The starting weights and the written weights are two files: a run that
    # starts from the warm start keeps that file intact.
    start_path = Path(args.init) if args.init else model_path
    if start_path.is_file():
        # The size of the network is stored with the weights, so a run that
        # continues an old one keeps its shape.
        model = load_model(str(start_path), device)
        print(f"读取权重: {start_path}", flush=True)
    else:
        model = GomokuNet(channels=args.channels, blocks=args.blocks).to(device)
        print(f"没有权重文件，从随机网络开始: {start_path}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer)
    if args.resume and args.buffer_file:
        buffer.load(args.buffer_file)
        print(f"读取回放缓冲区: {len(buffer.examples)} 个样本", flush=True)
    player = SelfPlay(
        model,
        device,
        c_puct=args.c_puct,
        simulations=args.simulations,
        candidate_radius=args.candidate_radius,
        temperature_moves=args.temperature_moves,
        seed=args.seed,
    )

    print(f"设备: {device}", flush=True)
    validation: list[Example] = []
    for iteration in range(1, args.iterations + 1):
        started = time.monotonic()
        wins = {1: 0, 2: 0}
        for _ in range(args.games):
            game = player.play_game()
            for example in game.examples:
                buffer.add(example)
            if game.winner is not None:
                wins[game.winner] += 1
        # The games of this iteration are the validation set of the next
        # round: they are trained on only after they were measured.
        validation = player.play_game().examples
        losses = train(
            model,
            optimizer,
            buffer,
            device,
            steps=args.train_steps,
            batch_size=args.batch_size,
            augment_data=not args.no_augment,
            rng=rng,
            log=True,
        )
        policy_loss, value_loss = evaluate(model, validation, device)
        save_checkpoint(model, model_path)
        if args.buffer_file:
            buffer.save(args.buffer_file)
        elapsed = time.monotonic() - started
        last = losses[-1] if losses else float("nan")
        print(
            f"第 {iteration} 轮: 胜负 {wins[1]}:{wins[2]}，"
            f"缓冲区 {len(buffer.examples)}，训练损失 {last:.4f}，"
            f"验证策略 {policy_loss:.4f} 验证价值 {value_loss:.4f}，"
            f"用时 {elapsed:.0f} 秒",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
