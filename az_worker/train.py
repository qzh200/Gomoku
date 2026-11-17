from __future__ import annotations

import argparse
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Event, Process, Queue
from pathlib import Path

import torch

from .dataset import TensorDataset
from .inference import serve
from .network import GomokuNet, load_model, read_extra, read_meta, train_batch
from .selfplay import (
    TEMPERATURE_MOVES,
    SelfPlay,
    checkpoint_stamp,
    save_checkpoint,
    worker_init,
    worker_play,
)


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

WORKER_VRAM = 256 * 1024 * 1024

WORKER_VRAM_RESERVE = 768 * 1024 * 1024


def plan_workers(
    requested,
    free_bytes: int | None,
    cpu_count: int,
    per_worker: int = WORKER_VRAM,
    reserve: int = WORKER_VRAM_RESERVE,
) -> int:
    if str(requested) != "auto":
        return max(1, int(requested))
    cores = max(1, cpu_count // 2)
    if free_bytes is None:
        return min(cores, 8)
    fits = max(1, (free_bytes - reserve) // per_worker)
    return max(1, min(cores, fits))


def plan_games(requested, workers: int) -> int:
    if str(requested) != "auto":
        return max(1, int(requested))
    return max(GAMES_PER_ITERATION, workers)


LOG_COLUMNS = (
    "time",
    "iteration",
    "step",
    "learning_rate",
    "simulations",
    "games",
    "total_games",
    "wins",
    "losses",
    "draws",
    "plies",
    "buffer",
    "policy_loss",
    "value_loss",
    "steps_per_second",
    "seconds",
    "eval_winrate",
)


def measure_strength(
    model_path,
    device,
    simulations: int,
    classical_depth: int,
    classical_time: float,
    games: int,
    seed: int,
    vcf_budget: float = 0.1,
    vct_budget: float = 0.1,
):
    from .arena import az_side, classical_side, play_series

    candidate = az_side(
        str(model_path),
        str(device),
        simulations,
        None,
        vcf_budget,
        vct_budget,
        seed=seed,
    )
    opponent = classical_side(classical_depth, 12, classical_time)
    return play_series(candidate, opponent, games, 2, seed)


def append_iteration_log(path: str | Path, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fresh = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as handle:
        if fresh:
            handle.write(",".join(LOG_COLUMNS) + "\n")
        handle.write(",".join(str(row[column]) for column in LOG_COLUMNS) + "\n")


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
    parser.add_argument(
        "--games",
        default="auto",
        help="每轮自对弈局数：auto 取进程数（池不空转）与 10 的较大者",
    )
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
        "--workers",
        default="auto",
        help="并行自对弈进程数：auto 按空闲显存与 CPU 核数推导，1 为串行，整数为固定值",
    )
    parser.add_argument(
        "--worker-device",
        default="auto",
        help="自对弈推理设备：auto 跟随训练设备；cpu 不抢显存但每局慢 4–5 倍；"
        "shared 让所有 CPU 自对弈进程共享一个 GPU 推理服务（显存只占一份）",
    )
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
    parser.add_argument(
        "--log-file",
        default="auto",
        help="每轮追加一行训练记录（CSV）；auto 写到权重同目录的 <model>.log，none 关闭",
    )
    parser.add_argument(
        "--tensorboard",
        default="auto",
        help="TensorBoard 事件目录；auto 写到权重同目录的 tensorboard/，none 关闭",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="每隔多少轮跟传统引擎打几局做棋力探针，0 关闭",
    )
    parser.add_argument("--eval-games", type=int, default=4, help="探针局数")
    parser.add_argument("--eval-simulations", type=int, default=128, help="探针每手模拟数")
    parser.add_argument("--eval-vcf", type=float, default=0.1, help="探针每手 VCF 预算")
    parser.add_argument("--eval-vct", type=float, default=0.1, help="探针每手 VCT 预算")
    parser.add_argument("--eval-classical-depth", type=int, default=2, help="探针对手深度")
    parser.add_argument("--eval-classical-time", type=float, default=0.05, help="探针对手每手秒数")
    parser.add_argument(
        "--progress",
        default="on",
        choices=("on", "off"),
        help="终端进度条（tqdm），off 用于日志重定向",
    )
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
    log_path = None
    if str(args.log_file).lower() not in ("", "none", "off"):
        log_path = (
            str(model_path.with_suffix(".log"))
            if str(args.log_file) == "auto"
            else args.log_file
        )
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
    games_done = 0
    dataset = TensorDataset(device, capacity=args.buffer)
    if args.resume and args.buffer_file:
        dataset = TensorDataset.load(args.buffer_file, device, args.buffer)
        if model_path.is_file():
            meta = read_meta(str(model_path))
            step = int(meta.get("step", 0))
            games_done = int(meta.get("games", 0))
            saved = read_extra(str(model_path)).get("optimizer")
            if saved:
                optimizer.load_state_dict(saved)
                print("恢复优化器状态（动量）", flush=True)
        print(
            f"续训: 缓冲区 {len(dataset)} 条，已完成 {step} 步、{games_done} 局",
            flush=True,
        )

    steps_per_iteration = max(1, math.ceil(args.total_steps / args.iterations))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    worker_device = args.worker_device
    if worker_device == "auto":
        worker_device = "cuda" if device.type == "cuda" else "cpu"
    free_bytes = None
    if worker_device == "cuda" and torch.cuda.is_available():
        free_bytes = int(torch.cuda.mem_get_info()[0])
    if worker_device == "shared":
        free_bytes = None
        if str(args.workers) == "auto":
            workers = max(2, (os.cpu_count() or 4) * 3 // 4)
        else:
            workers = max(1, int(args.workers))
    else:
        workers = plan_workers(args.workers, free_bytes, os.cpu_count() or 1)
    games = plan_games(args.games, workers)
    print(
        f"设备 {device}，参数 {parameters}，总步数 {args.total_steps}，"
        f"每轮 {steps_per_iteration} 步，轮数 {args.iterations}，每轮 {games} 局",
        flush=True,
    )
    if log_path:
        print(f"训练记录: {log_path}", flush=True)
    writer = None
    if str(args.tensorboard).lower() not in ("", "none", "off"):
        tb_dir = (
            str(model_path.parent / "tensorboard")
            if str(args.tensorboard) == "auto"
            else args.tensorboard
        )
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tb_dir)
        print(f"TensorBoard: {tb_dir}", flush=True)
    pool = None
    server = None
    stop_event = None
    requests = None
    if workers > 1:
        if worker_device == "shared":
            save_checkpoint(model, model_path, {"step": step})
            requests = Queue()
            stop_event = Event()
            server = Process(
                target=serve,
                args=(requests, str(device), str(model_path), stop_event),
                daemon=True,
            )
            server.start()
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(worker_device, args.temperature_moves, requests),
        )
        extra = "" if free_bytes is None else f"，空闲显存 {free_bytes / 2**20:.0f} MiB"
        print(
            f"并行自对弈: {workers} 个进程，推理设备 {worker_device}{extra}",
            flush=True,
        )

    from tqdm import tqdm

    progress = args.progress == "on"
    outer = tqdm(
        total=args.iterations,
        initial=min(args.iterations, step // steps_per_iteration),
        desc="训练",
        unit="轮",
        disable=not progress,
        position=0,
    )

    for iteration in range(1, args.iterations + 1):
        if step >= args.total_steps:
            break
        started = time.monotonic()
        step_started = step
        simulations = args.simulations or simulations_for(step, args.total_steps)
        if pool is not None:
            save_checkpoint(model, model_path, {"step": step, "games": games_done})
            stamp = checkpoint_stamp(str(model_path))
            tasks = [
                (
                    str(model_path),
                    stamp,
                    simulations,
                    args.search_batch,
                    rng.randrange(1 << 30),
                )
                for _ in range(games)
            ]
            futures = [pool.submit(worker_play, task) for task in tasks]
            games_played = [future.result() for future in futures]
        else:
            player = SelfPlay(
                model,
                device,
                simulations=simulations,
                batch_size=args.search_batch,
                temperature_moves=args.temperature_moves,
                seed=rng.randrange(1 << 30),
            )
            games_played = [player.play_game() for _ in range(games)]
        wins = {1: 0, 2: 0, None: 0}
        plies = 0
        for game in games_played:
            dataset.extend(game.examples)
            wins[game.winner] = wins.get(game.winner, 0) + 1
            plies += game.moves
        games_done += len(games_played)

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        bar = tqdm(
            total=steps_per_iteration,
            desc=f"第 {iteration} 轮",
            unit="步",
            leave=False,
            disable=not progress,
            position=1,
        )
        for _ in range(steps_per_iteration):
            if step >= args.total_steps:
                break
            current_lr = learning_rate(step, args.total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
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
            bar.update(1)
            bar.set_postfix(loss=f"{loss:.3f}", lr=f"{current_lr:.2e}")
            if writer is not None and step % 5 == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/policy_loss", policy_loss, step)
                writer.add_scalar("train/value_loss", value_loss, step)
                writer.add_scalar("train/learning_rate", current_lr, step)
        bar.close()

        save_checkpoint(
            model,
            model_path,
            {
                "step": step,
                "total_steps": args.total_steps,
                "iteration": iteration,
                "games": games_done,
                "lr": learning_rate(step, args.total_steps, args.lr),
                "simulations": simulations,
            },
            {"optimizer": optimizer.state_dict()},
        )
        if args.buffer_file:
            dataset.save(args.buffer_file)
        elapsed = time.monotonic() - started
        recent = len(policy_losses[-20:]) or 1
        mean_policy = sum(policy_losses[-20:]) / recent
        mean_value = sum(value_losses[-20:]) / recent
        per_second = (step - step_started) / max(elapsed, 1e-9)

        eval_winrate = ""
        eval_seconds = 0.0
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            probe_started = time.monotonic()
            try:
                probe = measure_strength(
                    model_path,
                    device,
                    args.eval_simulations,
                    args.eval_classical_depth,
                    args.eval_classical_time,
                    args.eval_games,
                    iteration,
                    args.eval_vcf,
                    args.eval_vct,
                )
                eval_winrate = f"{probe.winrate:.3f}"
                print(
                    f"棋力探针（第 {iteration} 轮）: {probe.wins} 胜 {probe.losses} 负 "
                    f"{probe.draws} 和，胜率 {probe.winrate:.0%}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("eval/winrate", probe.winrate, step)
                    writer.add_scalar("eval/games", probe.games, step)
            except Exception as error:  # 探针失败不影响训练
                print(f"棋力探针跳过: {error}", flush=True)
            eval_seconds = time.monotonic() - probe_started

        if writer is not None:
            writer.add_scalar("selfplay/plies", plies, step)
            writer.add_scalar("selfplay/simulations", simulations, step)
            writer.add_scalar("replay/buffer", len(dataset), step)
            writer.add_scalar("time/iteration_seconds", elapsed, step)
            writer.add_scalar("time/steps_per_second", per_second, step)
            writer.add_scalar("selfplay/games_total", games_done, step)
            writer.flush()

        if log_path:
            append_iteration_log(
                log_path,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "iteration": iteration,
                    "step": step,
                    "learning_rate": f"{learning_rate(step, args.total_steps, args.lr):.8g}",
                    "simulations": simulations,
                    "games": len(games_played),
                    "total_games": games_done,
                    "wins": wins[1],
                    "losses": wins[2],
                    "draws": wins[None],
                    "plies": plies,
                    "buffer": len(dataset),
                    "policy_loss": f"{mean_policy:.4f}",
                    "value_loss": f"{mean_value:.4f}",
                    "steps_per_second": f"{per_second:.2f}",
                    "seconds": f"{elapsed:.1f}",
                    "eval_winrate": eval_winrate,
                },
            )
        outer.update(1)
        outer.set_postfix(step=f"{step}/{args.total_steps}")
        print(
            f"第 {iteration} 轮: 步 {step}/{args.total_steps}，学习率 "
            f"{learning_rate(step, args.total_steps, args.lr):.6g}，模拟 {simulations}，"
            f"自对弈 {len(games_played)} 局（{wins[1]}:{wins[2]}，和 {wins[None]}，"
            f"{plies} 手），缓冲区 {len(dataset)}，"
            f"策略 {mean_policy:.4f} 价值 {mean_value:.4f}，"
            f"{per_second:.1f} 步/秒，用时 {elapsed:.0f} 秒"
            + (f"，探针 {eval_seconds:.0f} 秒" if eval_seconds else ""),
            flush=True,
        )
    outer.close()
    if writer is not None:
        writer.close()
    if pool is not None:
        pool.shutdown()
    if server is not None:
        stop_event.set()
        server.join(10.0)
        if server.is_alive():
            server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
