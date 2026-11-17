from __future__ import annotations

from . import board
from .openai_client import ChatMessage, OpenAIClient


SYSTEM_PROMPT = (
    "你在下一盘五子棋：横、竖、斜先连成五个棋子的一方获胜。"
    "棋盘 15 行 × 15 列，行号和列号都从 0 数到 14，坐标写成 (行, 列)。\n"
    "你要主动争胜：多造四、造活三，逼对手来堵，你先一步成五；"
    "对手要来成五时再回头挡。只守不攻赢不了，能进攻就去进攻。\n"
    "「成五点」：这一手落下去立刻连成五个子，只有成五点才等于赢。\n"
    "「活四点」：落下去以后才有两个成五点，要等下一手才赢——对方会先走，慢一手就是输。\n"
    "「冲四点」：落下去以后只有一个成五点，对手必须去堵，这是抢攻的最好办法。\n"
    "下面会给你双方全部棋子的坐标、双方手里这三种点，以及对手的最后一手。"
    "你要为「我方」下一个空点。按这个顺序判断：\n"
    "1. 我方有成五点：下在那里，立刻赢。\n"
    "2. 对方有成五点：这一手只能去堵它，别下别的——你的活四慢一手，来不及。\n"
    "3. 我方有成活四的点：下那里，对方挡不住，下一手就赢。\n"
    "4. 我方有成冲四的点，对方没有成五：下那里抢攻，对手只能来堵，主动权在你。\n"
    "5. 对方有成活四的点：现在就得挡，否则下一手挡不住。\n"
    "6. 都没有：主动进攻，下在能把自己的两子接成三子、三子接成四子的点，"
    "靠近对方棋子，别下边角。\n"
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
            move = board.parse_move(answer.text)
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

    def _messages(
        self,
        board_state: list[list[int]],
        player: int,
        last_move: dict | None,
    ) -> list[ChatMessage]:
        opponent = board.opponent_of(player)
        mine = board.threats(board_state, player)
        theirs = board.threats(board_state, opponent)
        lines = [
            f"棋盘是 {board.BOARD_SIZE} 行 × {board.BOARD_SIZE} 列，"
            f"行号和列号都从 0 数到 {board.BOARD_SIZE - 1}。",
        ]
        lines.extend(_this_move(mine, theirs, player))
        lines.append(
            f"我方（{board.player_name(player)}）的棋子坐标："
            + board.stone_list(board_state, player)
        )
        lines.append(
            f"对方（{board.player_name(opponent)}）的棋子坐标："
            + board.stone_list(board_state, opponent)
        )
        lines.append("对方的棋形：")
        lines.extend(_threat_lines(theirs, opponent))
        lines.append("我方的棋形：")
        lines.extend(_threat_lines(mine, player))
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


def _this_move(mine: board.Threats, theirs: board.Threats, player: int) -> list[str]:
    """这一手该怎么下，由成五点/活四点决定，写成一句话。"""
    name = board.player_name(player)
    if mine.fives:
        return [f"这一手：{name}下在 {board.spots(mine.fives)} 就成五，立刻获胜。"]
    if len(theirs.fives) == 1:
        return [
            f"这一手：对方下在 {board.spots(theirs.fives)} 就成五，"
            "必须堵这里，下别处下一手就输。"
        ]
    if theirs.fives:
        return [
            f"这一手：对方有 {len(theirs.fives)} 个成五点 "
            f"{board.spots(theirs.fives)}，堵一个也挡不住，只能自己立刻成五。"
        ]
    if mine.open_fours:
        return [
            f"这一手：{name}下在 {board.spots(mine.open_fours)} 就成活四，对方挡不住。"
        ]
    if theirs.open_fours:
        return [
            f"这一手：对方下在 {board.spots(theirs.open_fours)} 就成活四，必须现在处理。"
        ]
    return []


def _threat_lines(found: board.Threats, player: int) -> list[str]:
    lines = board.describe(found, player)
    if not lines:
        return ["- 没有成五、活四或冲四的落点。"]
    return [f"- {line}" for line in lines]
