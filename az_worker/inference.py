from __future__ import annotations

import time

import numpy as np
import torch
from multiprocessing.connection import wait

from .network import evaluate, load_model


DEFAULT_MAX_BATCH = 192

DEFAULT_WINDOW = 0.002

DEFAULT_TIMEOUT = 120.0

RELOAD_CHECK = 0.25


class RemoteEvaluator:
    def __init__(self, connection, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.connection = connection
        self.timeout = timeout

    def __call__(self, planes: np.ndarray, masks: np.ndarray):
        self.connection.send((planes.astype(np.float16), masks.astype(np.float16)))
        if not self.connection.poll(self.timeout):
            raise RuntimeError("推理服务没有响应，检查共享推理进程是否退出")
        policies, values = self.connection.recv()
        return policies, values


def evaluate_remote(
    connection,
    planes: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return RemoteEvaluator(connection)(planes, masks)


def serve(
    requests,
    device: str,
    model_path: str,
    stop_event,
    max_batch: int = DEFAULT_MAX_BATCH,
    window: float = DEFAULT_WINDOW,
) -> None:
    target = torch.device(device)
    model = load_model(model_path, target)
    model.eval()
    stamp = _stamp(model_path)
    last_check = time.monotonic()
    alive: list = []

    while not stop_event.is_set():
        while True:
            try:
                alive.append(requests.get_nowait())
            except Exception:
                break
        if not alive:
            time.sleep(window)
            continue

        ready = wait(alive, timeout=window)
        if not ready:
            continue

        batch: list[tuple[object, tuple[np.ndarray, np.ndarray]]] = []
        for connection in ready:
            message = _receive(connection)
            if message is None:
                alive.remove(connection)
                continue
            batch.append((connection, message))
        _drain(batch, alive, wait, max_batch)

        if not batch:
            continue

        now = time.monotonic()
        if now - last_check > RELOAD_CHECK:
            last_check = now
            fresh = _stamp(model_path)
            if fresh != stamp:
                model = load_model(model_path, target)
                model.eval()
                stamp = fresh

        planes = np.concatenate([message[0] for _, message in batch]).astype(np.float32)
        masks = np.concatenate([message[1] for _, message in batch]).astype(np.float32)
        policies, values = evaluate(model, planes, masks, target)
        start = 0
        for connection, message in batch:
            size = len(message[0])
            try:
                connection.send((policies[start : start + size], values[start : start + size]))
            except (BrokenPipeError, OSError):
                pass
            start += size


def _receive(connection):
    try:
        return connection.recv()
    except (EOFError, OSError):
        return None


def _drain(batch, alive, waiter, max_batch: int) -> None:
    while len(batch) < max_batch and alive:
        more = waiter(alive, timeout=0)
        if not more:
            break
        for connection in more:
            message = _receive(connection)
            if message is None:
                alive.remove(connection)
                continue
            batch.append((connection, message))


def _stamp(path: str) -> int:
    import os

    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0
