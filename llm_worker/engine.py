"""The LLM engine: it asks the model for a move and checks the answer.

The engine holds no state of the game. It reads the board of a turn, asks the
model where to play and returns the move it picked. The protocol of the game
lives in the worker, the model in the client, so both can be replaced alone.
"""

from __future__ import annotations

import json

from . import board
from .openai_client import APIError, ChatMessage, OpenAIClient

# The system prompt tells the model the rules of the game and the shape of
# the answer. The board in the user message carries the position.
SYSTEM_PROMPT = (
    "你是五子棋引擎，棋盘为 15×15，黑棋先行。"
    "你收到棋盘、落子历史和当前执子方，请选择一个空点落子，形成五连或为后续连五做准备。"
    "只输出 JSON：{\"row\": <行号 0-14>, \"col\": <列号 0-14>}，不要输出其他内容。"
)

# The error a move gets when it is not on the board or on a free cell.
INVALID_MOVE_HINT = "该位置无效或已有棋子，请换一个空点。"


class EngineError(RuntimeError):
    """The model did not answer with a playable move."""


class LLMEngine:
    """Picks a move with a language model."""

    def __init__(self, client: OpenAIClient, max_attempts: int = 2) -> None:
        self.client = client
        self.max_attempts = max(1, max_attempts)

    def choose_move(
        self,
        board_state: list[list[int]],
        history: list[dict],
        player: int,
        last_move: dict | None = None,
        config: dict | None = None,
    ) -> tuple[int, int]:
        """Return the row and the column of the move the model picked."""
        messages = self._messages(board_state, history, player, last_move, config)
        problem = ""
        for attempt in range(self.max_attempts):
            answer = self.client.complete(messages)
            move = board.parse_move(answer)
            if move is None:
                problem = "回答里没有可解析的落点，请只输出 JSON。"
                messages.append(ChatMessage("assistant", answer))
                messages.append(ChatMessage("user", problem))
                continue
            row, col = move
            if not board.is_empty(board_state, row, col):
                problem = INVALID_MOVE_HINT
                messages.append(ChatMessage("assistant", answer))
                messages.append(ChatMessage("user", problem))
                continue
            return row, col
        raise EngineError(problem or "模型没有给出合法落点")

    def _messages(
        self,
        board_state: list[list[int]],
        history: list[dict],
        player: int,
        last_move: dict | None,
        config: dict | None,
    ) -> list[ChatMessage]:
        """Build the conversation that asks for a move."""
        lines = [
            "当前棋盘（列 A-O，行 1-15，X 是黑棋，O 是白棋，. 是空点）：",
            board.render_board(board_state),
            "",
            "落子历史：",
            board.render_history(history),
            "",
            f"当前执子方：{board.player_name(player)}",
        ]
        if last_move:
            coordinate = board.coordinate_of(last_move["row"], last_move["col"])
            lines.append(f"最后一手：{coordinate}（{board.player_name(last_move['player'])}）")
        if config:
            lines.append("")
            lines.append("引擎参数：" + json.dumps(config, ensure_ascii=False))
        lines.append("")
        lines.append("请给出下一步落点的 JSON。")
        return [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", "\n".join(lines)),
        ]
