#!/usr/bin/env python3
"""Mock MyGo AI worker.

A reference implementation of the Python worker protocol: it long polls for a
turn, picks a legal move with a simple heuristic and submits it. It is used to
verify that a worker can be written against the frozen contract alone.

Usage:
    python mock_worker.py --engine classical --token <engine_token>
    python mock_worker.py --engine classical --token <engine_token> \
        --url http://127.0.0.1:8080/api/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

BOARD_SIZE = 15


def request(url: str, method: str, token: str | None = None, body: Any = None) -> tuple[int, bytes]:
    """Send one request and return the status and the body."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def score_line(board: list[list[int]], row: int, col: int, player: int) -> int:
    """Score the line through a cell for the given player."""
    total = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                r += dr * sign
                c += dc * sign
        if count >= 5:
            total += 1_000_000
        else:
            total += count * count
    return total


def pick_move(board: list[list[int]], player: int) -> tuple[int, int]:
    """Pick the best empty cell for the given player."""
    best: tuple[int, int] | None = None
    best_score = -1
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != 0:
                continue
            score = score_line(board, row, col, player)
            score += score_line(board, row, col, 3 - player) // 2
            # Prefer the center a little.
            score += max(0, 7 - abs(row - 7) - abs(col - 7))
            if score > best_score:
                best_score = score
                best = (row, col)
    if best is None:
        raise RuntimeError("棋盘已满，没有可落子的位置")
    return best


def run(url: str, engine: str, token: str, worker_id: str) -> int:
    """Poll for turns until the process is stopped."""
    print(f"[{worker_id}] 开始轮询 {url}", flush=True)
    while True:
        status, body = request(
            f"{url}/ai/turn-claims",
            "POST",
            token,
            {"worker_id": worker_id, "wait_seconds": 20},
        )
        if status == 204:
            continue
        if status != 201:
            print(f"[{worker_id}] 领取任务失败: {status} {body.decode('utf-8', 'replace')}", file=sys.stderr)
            time.sleep(2)
            continue

        claim = json.loads(body)
        game_id = claim["game_id"]
        version = claim["version"]
        claim_id = claim["claim_id"]
        player = claim["player"]
        board = claim["board"]
        row, col = pick_move(board, player)
        print(f"[{worker_id}] 领取 {claim['turn_id']}，落子 ({row},{col})", flush=True)

        status, body = request_with_key(
            f"{url}/games/{game_id}/moves",
            token,
            {
                "row": row,
                "col": col,
                "expected_version": version,
                "claim_id": claim_id,
            },
        )
        if status == 201:
            print(f"[{worker_id}] 落子成功", flush=True)
            continue
        print(f"[{worker_id}] 落子被拒绝: {status} {body.decode('utf-8', 'replace')}", file=sys.stderr)
        time.sleep(0.5)


def request_with_key(url: str, token: str, body: dict[str, Any]) -> tuple[int, bytes]:
    """Send a move with an idempotency key."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Idempotency-Key", str(uuid.uuid4()))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="MyGo mock AI worker")
    parser.add_argument("--url", default="http://127.0.0.1:8080/api/v1", help="MyGo API 根地址")
    parser.add_argument("--engine", required=True, help="引擎标识，仅用于日志")
    parser.add_argument("--token", required=True, help="引擎 Bearer Token")
    parser.add_argument("--worker-id", default=None, help="Worker 标识")
    args = parser.parse_args()
    worker_id = args.worker_id or f"{args.engine}-worker-01"
    try:
        return run(args.url, args.engine, args.token, worker_id)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
