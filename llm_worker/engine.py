"""The LLM engine: it asks the model for a move and checks the answer.

The engine holds no state of the game. It reads the board of a turn, asks the
model where to play and returns the move it picked. The protocol of the game
lives in the worker, the model in the client, so both can be replaced alone.
"""

from __future__ import annotations

import json

from . import board
from .openai_client import Answer, ChatMessage, OpenAIClient

# The system prompt tells the model the rules of the game and the shape of
# the answer. The board in the user message carries the position.
SYSTEM_PROMPT = (
    "你是五子棋引擎，棋盘 15×15，黑棋先行，横竖斜连续五子获胜。"
    "你收到棋盘、落子历史和当前执子方，请选一个空点落子。"
    "优先自己连成五子；对手有四子连成一线时，必须先堵住它。"
    "坐标就用棋盘上的标注：列是 A-O，行是 1-15。"
    '只输出 JSON：{"move": "M5"}，不要输出其他内容。'
)

# The problem a move gets when it is not on the board or on a free cell.
INVALID_MOVE_HINT = "该位置无效或已有棋子，请换一个空点。"

# The reminder of the shape of the answer, after an answer that had no move.
FORMAT_HINT = '请只输出 JSON：{"move": "M5"}，坐标用棋盘的列字母和行号。'


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
        offered = set(board.candidates(board_state))
        problem = ""
        for _ in range(self.max_attempts):
            answer = self.client.complete(messages)
            move = self._move_of(answer, board_state)
            if move is None:
                problem = FORMAT_HINT
                messages.append(ChatMessage("assistant", answer.text))
                messages.append(ChatMessage("user", problem))
                continue
            row, col = move
            if not board.is_empty(board_state, row, col):
                # Naming the cell it picked tells the model what went wrong.
                problem = f"{board.coordinate_of(row, col)} {INVALID_MOVE_HINT}"
                messages.append(ChatMessage("assistant", answer.text))
                messages.append(ChatMessage("user", problem))
                continue
            if move not in offered:
                # A stone far from the others helps nobody; the list of the
                # cells worth playing keeps the answer in the fight.
                problem = f"{board.coordinate_of(row, col)} 离战场太远，请从空点列表里选一个。"
                messages.append(ChatMessage("assistant", answer.text))
                messages.append(ChatMessage("user", problem))
                continue
            return row, col
        raise EngineError(problem or "模型没有给出合法落点")

    def _move_of(self, answer: "Answer", board_state: list[list[int]]) -> tuple[int, int] | None:
        """Read the move out of the answer of the model.

        A reasoning model that ran out of tokens answers in its reasoning
        text, where a coordinate may be mentioned in passing. Only a JSON
        object is trusted there; the plain content may also be a coordinate.
        """
        if answer.from_reasoning:
            return board.parse_move_strict(answer.text)
        return board.parse_move(answer.text)

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
        free = [board.coordinate_of(row, col) for row, col in board.candidates(board_state)]
        lines.append("")
        lines.append("只能下在这些空点之一：" + " ".join(free))
        warnings = board.threats(board_state, player)
        if warnings:
            lines.append("")
            lines.append("必须先处理对手的连子：")
            lines.extend(warnings)
        if config:
            lines.append("")
            lines.append("引擎参数：" + json.dumps(config, ensure_ascii=False))
        lines.append("")
        lines.append('请从上面的空点里选一个，用棋盘坐标回答，例如 {"move": "M5"}。')
        return [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", "\n".join(lines)),
        ]
