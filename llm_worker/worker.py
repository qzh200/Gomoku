"""The worker: it plays the turns the server hands out.

The worker long polls for a turn, asks the engine for a move and submits it.
It renews the lease while the model thinks, so a slow model does not lose
its turn.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .engine import LLMEngine


class WorkerError(RuntimeError):
    """The server refused a request of the worker."""


class Claim:
    """One turn of a game, handed out by the server."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.claim_id: str = payload["claim_id"]
        self.turn_id: str = payload.get("turn_id", "")
        self.game_id: str = payload["game_id"]
        self.version: int = payload["version"]
        self.player: int = payload["player"]
        self.board: list[list[int]] = payload["board"]
        self.history: list[dict] = payload.get("history", [])
        self.last_move: dict | None = payload.get("last_move")
        self.config: dict = payload.get("config") or {}

    @property
    def expires_at(self) -> str:
        """The time the lease runs out, as the server wrote it."""
        return self.payload.get("expires_at", "")


class MyGoClient:
    """The client of the MyGo API the worker talks to."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float = 180.0,
    ) -> tuple[int, bytes]:
        """Send one request and return the status and the body."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.token}")
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def claim(self, worker_id: str, wait_seconds: int) -> Claim | None:
        """Long poll for a turn. None means there was no turn to play."""
        status, body = self.request(
            "POST",
            "/ai/turn-claims",
            {"worker_id": worker_id, "wait_seconds": wait_seconds},
        )
        if status == 204:
            return None
        if status != 201:
            raise WorkerError(f"领取任务失败: {status} {body.decode('utf-8', 'replace')}")
        return Claim(json.loads(body))

    def renew(self, claim_id: str, extend_seconds: int) -> bool:
        """Extend the lease of a turn."""
        status, body = self.request(
            "PATCH",
            f"/ai/turn-claims/{claim_id}",
            {"extend_seconds": extend_seconds},
        )
        if status != 200:
            raise WorkerError(f"续期租约失败: {status} {body.decode('utf-8', 'replace')}")
        return True

    def release(self, claim_id: str) -> None:
        """Give the turn back to the server."""
        self.request("DELETE", f"/ai/turn-claims/{claim_id}")

    def submit_move(
        self,
        game_id: str,
        row: int,
        col: int,
        expected_version: int,
        claim_id: str,
        idempotency_key: str,
    ) -> tuple[int, bytes]:
        """Play a move. The status tells whether the server took it."""
        return self.request(
            "POST",
            f"/games/{game_id}/moves",
            {
                "row": row,
                "col": col,
                "expected_version": expected_version,
                "claim_id": claim_id,
            },
            headers={"Idempotency-Key": idempotency_key},
            timeout=60.0,
        )


class LeaseKeeper:
    """Renews the lease of a turn while the model thinks.

    The renewal runs in a thread, so a long answer does not lose the turn.
    """

    def __init__(self, client: MyGoClient, claim_id: str, extend_seconds: int) -> None:
        self.client = client
        self.claim_id = claim_id
        self.extend_seconds = extend_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LeaseKeeper":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        """Renew the lease until the turn is over."""
        interval = max(5.0, self.extend_seconds / 3)
        while not self._stop.wait(interval):
            try:
                self.client.renew(self.claim_id, self.extend_seconds)
            except WorkerError:
                return


class Worker:
    """Plays the turns of the games the server hands out."""

    def __init__(
        self,
        client: MyGoClient,
        engine: LLMEngine,
        worker_id: str,
        wait_seconds: int = 20,
        extend_seconds: int = 120,
        backoff: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.client = client
        self.engine = engine
        self.worker_id = worker_id
        self.wait_seconds = wait_seconds
        self.extend_seconds = extend_seconds
        # The delay after a failure grows, so a broken model service is not
        # asked over and over.
        self.backoff = backoff
        self.max_backoff = max_backoff

    def run(self) -> None:
        """Poll for turns until the process is stopped.

        A failure does not stop the worker, but it does slow it down: the
        same turn would be handed out again right away, and a broken model
        service would be asked over and over.
        """
        print(f"[{self.worker_id}] 开始轮询 {self.client.base_url}", flush=True)
        failures = 0
        while True:
            try:
                played = self.play_turn()
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - the loop must survive
                failures += 1
                delay = min(self.max_backoff, self.backoff * failures)
                print(
                    f"[{self.worker_id}] 本轮失败: {error}；{delay:.0f} 秒后重试",
                    flush=True,
                )
                time.sleep(delay)
                continue
            if played:
                failures = 0
            else:
                time.sleep(self.backoff)

    def play_turn(self) -> bool:
        """Play one turn. False means there was no turn to play."""
        claim = self.client.claim(self.worker_id, self.wait_seconds)
        if claim is None:
            return False
        print(f"[{self.worker_id}] 领取 {claim.turn_id}", flush=True)
        try:
            with LeaseKeeper(self.client, claim.claim_id, self.extend_seconds):
                row, col = self.engine.choose_move(
                    claim.board,
                    claim.history,
                    claim.player,
                    claim.last_move,
                    claim.config,
                )
        except Exception:
            # The turn cannot be played, so it goes back to the server.
            self.client.release(claim.claim_id)
            raise
        print(f"[{self.worker_id}] 落子 ({row},{col})", flush=True)

        status, body = self.client.submit_move(
            claim.game_id,
            row,
            col,
            claim.version,
            claim.claim_id,
            str(uuid.uuid4()),
        )
        if status == 201:
            print(f"[{self.worker_id}] 落子成功", flush=True)
            return True
        detail = body.decode("utf-8", "replace")
        if status == 409:
            # The turn changed under the worker; the result is dropped.
            print(f"[{self.worker_id}] 落子已过期，丢弃结果: {detail}", flush=True)
            return False
        raise WorkerError(f"落子被拒绝: {status} {detail}")
