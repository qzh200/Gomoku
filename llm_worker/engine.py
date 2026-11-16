from __future__ import annotations

from . import board
from .openai_client import Answer, ChatMessage, OpenAIClient


SYSTEM_PROMPT = (
    "你是五子棋引擎，棋盘 15×15，黑棋先行，横竖斜连续五子获胜。"
    "这是一场你必须赢的对局。你的目标是连成五子赢棋，不是只挡住对方——"
    "只防守赢不了，必须自己造出威胁。\n"
    "坐标用 (行, 列)，行和列都从 0 数到 14。\n"
    "按下面的顺序判断这一手，前面的优先级更高：\n"
    "1. 能直接连成五子，就下那一点，立刻赢。\n"
    "2. 对方已经四子连成一线时，这一手必须堵住它的延伸端，"
    "否则对方下一手就五连赢了。\n"
    "3. 对方有三子连成一线、两头都能接上（活三）时，必须堵住一头，"
    "否则对方下一手就成活四，你挡不住。\n"
    "4. 没有紧急威胁就主动进攻：把自己的三子接成四子。"
    "四子是极强的武器——对方必须来堵，堵完你就再长一子成五连。"
    "把对手逼着走，比自己闷头下棋更容易赢。\n"
    "5. 什么都没有的时候，把自己的棋子摆成一条线，"
    "下在靠近对方棋子、也靠近自己棋子的空点上，别下边角。\n"
    "永远先问自己「这一手能让我连成五子吗」，再问「对方下一手能赢吗」。"
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
        messages = self._messages(board_state, history, player, last_move, config)
        problem = ""
        for _ in range(self.max_attempts):
            answer = self.client.complete(messages)
            move = self._move_of(answer, board_state)
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

    def _move_of(self, answer: "Answer", board_state: list[list[int]]) -> tuple[int, int] | None:
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
        opponent = board.opponent_of(player)
        lines = [
            f"棋盘是 {board.BOARD_SIZE} 行 × {board.BOARD_SIZE} 列，"
            "行号和列号都从 0 数到 14。",
            "",
            "当前棋盘（行号在前，每行 15 个格子，空格是 .，X 是黑棋，O 是白棋）：",
            board.render_board(board_state),
            "",
            f"我方（{board.player_name(player)}）的棋子坐标："
            + board.stone_list(board_state, player),
            f"对方（{board.player_name(opponent)}）的棋子坐标："
            + board.stone_list(board_state, opponent),
            "",
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
        if history:
            lines.append("")
            lines.append("落子历史：")
            lines.append(board.render_history(history))
        lines.append(f"当前轮到你执{board.player_name(player)}。")
        if config:
            lines.append("")
            lines.append("引擎参数：" + str(config))
        lines.append("")
        lines.append("这一手的选择：")
        lines.append(self._plan(board_state, player, opponent))
        lines.append("")
        lines.append(
            "请选一个空点落子，回答格式："
            '{"move": {"row": 行号, "col": 列号}}，例如 {"move": {"row": 7, "col": 7}}。'
        )
        return [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", "\n".join(lines)),
        ]

    def _plan(
        self, board_state: list[list[int]], player: int, opponent: int
    ) -> str:
        """把这一手的关键选择写成文字：先说赢棋点，再说挡点，最后说进攻点。"""
        notes: list[str] = []

        wins = self._winning_cells(board_state, player)
        if wins:
            notes.append(
                "1. 你现在就能连成五子获胜，直接下在 "
                + "、".join(f"({r}, {c})" for r, c in wins)
                + "。"
            )

        blocks = self._must_block(board_state, opponent)
        if blocks:
            notes.append(
                "2. 【必须挡这里】对方下一手就能连成五子，"
                "这一手只能下在 "
                + "、".join(f"({r}, {c})" for r, c in blocks)
                + "，下别的地方你马上就输。"
            )

        threats = self._open_threes(board_state, opponent)
        if threats:
            notes.append(
                "3. 对方有三子连成一线而且两头都能接上（活三），"
                "只堵一头挡不住，你必须下在 "
                + "、".join(f"({r}, {c})" for r, c in threats)
                + " 里的一个，或者抢先连成四子。"
            )

        attacks = self._attack_cells(board_state, player)
        if attacks:
            notes.append(
                "4. 你可以在这里把自己的棋子接成四子，抢占主动："
                + "、".join(f"({r}, {c})" for r, c in attacks)
                + "。"
            )

        builds = self._build_cells(board_state, player)
        if builds:
            notes.append(
                "5. 想从零开始布阵时，可以把棋子接成三子："
                + "、".join(f"({r}, {c})" for r, c in builds)
                + "。"
            )

        if not notes:
            notes.append(
                "目前没有紧急威胁。选一个靠近对方棋子、也靠近自己棋子的空点，"
                "把自己的棋子摆成一条线，不要下在边角空地。"
            )
        return "\n".join(notes)

    def _winning_cells(
        self, board_state: list[list[int]], player: int
    ) -> list[tuple[int, int]]:
        cells = []
        for run, free in board.connected_lines(board_state, player, 4):
            if len(run) == 4:
                cells.extend(free)
        return cells

    def _must_block(
        self, board_state: list[list[int]], opponent: int
    ) -> list[tuple[int, int]]:
        cells = []
        for run, free in board.connected_lines(board_state, opponent, 4):
            if len(run) == 4:
                cells.extend(free)
        return cells

    def _open_threes(
        self, board_state: list[list[int]], opponent: int
    ) -> list[tuple[int, int]]:
        cells = []
        for run, free in board.connected_lines(board_state, opponent, 3):
            if len(run) == 3 and len(free) == 2:
                cells.extend(free)
        return cells

    def _attack_cells(
        self, board_state: list[list[int]], player: int
    ) -> list[tuple[int, int]]:
        cells = []
        for run, free in board.connected_lines(board_state, player, 3):
            if len(run) == 3 and len(free) == 2:
                cells.extend(free)
        return cells

    def _build_cells(
        self, board_state: list[list[int]], player: int
    ) -> list[tuple[int, int]]:
        cells = []
        for run, free in board.connected_lines(board_state, player, 2):
            if len(run) == 2 and free:
                cells.extend(free)
        return cells
