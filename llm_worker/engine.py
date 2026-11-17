from __future__ import annotations

from . import board
from .openai_client import Answer, ChatMessage, OpenAIClient


SYSTEM_PROMPT = (
    "你在下一盘五子棋：横、竖、斜先连成五个棋子的一方获胜。"
    "棋盘 15 行 × 15 列，行号和列号都从 0 数到 14，坐标写成 (行, 列)。\n"
    "下面会给你「我方」和「对方」全部棋子的坐标，以及对手的最后一手。"
    "你要为「我方」下一个空点。每一手都按这个顺序判断：\n"
    "1. 对手下一手就能连成五个子吗？能，这一手必须先挡住，下别处立刻输。\n"
    "2. 自己这一手就能连成五个子吗？能，就下在那里，直接赢。\n"
    "3. 对手有三子连成一线、两头都是空点吗？必须堵住一头。\n"
    "4. 都不是，就把自己的三子接成四子，或者下在自己棋子旁边的空点。\n"
    '只输出 JSON：{"move": {"row": 7, "col": 7}}，不要输出其他内容。'
)


INVALID_MOVE_HINT = "该位置无效或已有棋子，请换一个空点。"


FORMAT_HINT = (
    '请只输出 JSON：{"move": {"row": 7, "col": 7}}，'
    "row 和 col 是 0 到 14 之间的整数。"
)


class EngineError(RuntimeError):
    pass


class LLMEngine:
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
        if board.fallback_cell(board_state) is None:
            raise EngineError("棋盘已满，没有可落子的位置")
        messages = self._messages(board_state, player, last_move)
        problem = ""
        for _ in range(self.max_attempts):
            answer = self.client.complete(messages)
            move = self._move_of(answer)
            if move is None:
                problem = FORMAT_HINT
            else:
                row, col = move
                if not board.is_empty(board_state, row, col):
                    problem = f"({row}, {col}) {INVALID_MOVE_HINT}"
                else:
                    return row, col
            messages.append(ChatMessage("assistant", answer.text))
            messages.append(ChatMessage("user", problem))

        fallback = board.fallback_cell(board_state)
        if fallback is None:
            raise EngineError("棋盘已满，没有可落子的位置")
        print(
            f"[llm] 模型 {self.max_attempts} 次都没给出空点，改用 "
            f"({fallback[0]}, {fallback[1]})",
            flush=True,
        )
        return fallback

    def _move_of(self, answer: "Answer") -> tuple[int, int] | None:
        if answer.from_reasoning:
            return board.parse_move_strict(answer.text)
        return board.parse_move(answer.text)

    def _messages(
        self,
        board_state: list[list[int]],
        player: int,
        last_move: dict | None,
    ) -> list[ChatMessage]:
        opponent = board.opponent_of(player)
        lines = [
            f"棋盘是 {board.BOARD_SIZE} 行 × {board.BOARD_SIZE} 列，"
            f"行号和列号都从 0 数到 {board.BOARD_SIZE - 1}。",
            f"我方（{board.player_name(player)}）的棋子坐标："
            + board.stone_list(board_state, player),
            f"对方（{board.player_name(opponent)}）的棋子坐标："
            + board.stone_list(board_state, opponent),
            board.describe_lines(board_state, player),
            board.describe_lines(board_state, opponent),
        ]
        if last_move:
            side = board.player_name(last_move["player"])
            lines.append(
                f"最后一手是{side}，下在 ({last_move['row']}, {last_move['col']})。"
            )
        else:
            lines.append("这是第一手。")
        lines.append(
            "请给一个空点落子，只输出 "
            '{"move": {"row": 行号, "col": 列号}}。'
        )
        return [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", "\n".join(lines)),
        ]
