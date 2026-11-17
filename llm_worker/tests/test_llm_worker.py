from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_worker import board
from aiworker.settings import ConfigError, parse_dotenv, read_runtime_token
from llm_worker.config import Settings
from llm_worker.engine import EngineError, LLMEngine
from llm_worker.openai_client import APIError, Answer, ChatMessage, OpenAIClient


class FakeClient:
    def __init__(self, answers: list[str | tuple[str, bool]]) -> None:
        self.answers = list(answers)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> Answer:
        self.calls.append(messages)
        if not self.answers:
            raise APIError("没有更多回答")
        answer = self.answers.pop(0)
        if isinstance(answer, tuple):
            return Answer(answer[0], from_reasoning=answer[1])
        return Answer(answer)


def empty_board() -> list[list[int]]:
    return [[0] * board.BOARD_SIZE for _ in range(board.BOARD_SIZE)]


def board_state_value(state: list[list[int]], cell: tuple[int, int]) -> int:
    row, col = cell
    return state[row][col]


class ParseDotenvTests(unittest.TestCase):
    def test_reads_pairs_and_skips_comments(self) -> None:
        values = parse_dotenv(
            "\n".join(
                [
                    "# a comment",
                    "",
                    "OPENAI_MODEL=gpt-4o-mini",
                    'OPENAI_API_KEY="sk-secret"',
                    "export OPENAI_TEMPERATURE=0.4",
                    "OPENAI_MAX_TOKENS=200  ",
                ]
            )
        )
        self.assertEqual(values["OPENAI_MODEL"], "gpt-4o-mini")
        self.assertEqual(values["OPENAI_API_KEY"], "sk-secret")
        self.assertEqual(values["OPENAI_TEMPERATURE"], "0.4")
        self.assertEqual(values["OPENAI_MAX_TOKENS"], "200")

    def test_rejects_a_line_without_a_value(self) -> None:
        with self.assertRaises(ConfigError):
            parse_dotenv("OPENAI_MODEL")


class SettingsTests(unittest.TestCase):
    def test_reads_the_environment(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_URL": "http://127.0.0.1:9000/api/v1/",
                "MYGO_ENGINE_TOKEN": "token",
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "deepseek-chat",
                "OPENAI_TEMPERATURE": "0.3",
                "OPENAI_MAX_TOKENS": "128",
            }
        )
        self.assertEqual(settings.mygo_url, "http://127.0.0.1:9000/api/v1")
        self.assertEqual(settings.engine_token, "token")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertEqual(settings.temperature, 0.3)
        self.assertEqual(settings.max_tokens, 128)
        self.assertEqual(settings.worker_id, "llm-worker-01")

    def test_reports_a_missing_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.runtime.json"
            path.write_text(json.dumps({"engines": []}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Settings.from_env(
                    {
                        "MYGO_RUNTIME_CONFIG": str(path),
                        "OPENAI_API_KEY": "key",
                    }
                )

    def test_reports_a_number_that_is_not_a_number(self) -> None:
        with self.assertRaises(ConfigError):
            Settings.from_env(
                {
                    "MYGO_ENGINE_TOKEN": "token",
                    "OPENAI_API_KEY": "key",
                    "OPENAI_TEMPERATURE": "warm",
                }
            )


class BoardTests(unittest.TestCase):
    def test_lists_the_stones_of_one_side(self) -> None:
        state = empty_board()
        state[0][0] = board.BLACK
        state[1][2] = board.WHITE
        self.assertEqual(board.stone_list(state, board.BLACK), "(0, 0)")
        self.assertEqual(board.stone_list(state, board.WHITE), "(1, 2)")

    def test_reports_the_two_five_points_of_an_open_four(self) -> None:
        state = empty_board()
        for col in range(7, 11):
            state[7][col] = board.BLACK
        self.assertEqual(board.five_points(state, board.BLACK), [(7, 6), (7, 11)])
        line = board.describe_threats(state, board.BLACK)[0]
        self.assertIn("2 个成五点", line)
        self.assertIn("堵一个也挡不住", line)

    def test_reports_the_single_five_point_of_a_blocked_four(self) -> None:
        state = empty_board()
        for col in range(7, 11):
            state[7][col] = board.BLACK
        state[7][6] = board.WHITE
        self.assertEqual(board.five_points(state, board.BLACK), [(7, 11)])
        self.assertEqual(
            board.describe_threats(state, board.BLACK),
            ["黑棋下在 (7, 11) 就成五。"],
        )

    def test_calls_an_open_three_an_open_four_maker(self) -> None:
        state = empty_board()
        for row, col in ((7, 7), (6, 8), (5, 9)):
            state[row][col] = board.BLACK
        open_fours, fours = board.four_makers(state, board.BLACK)
        self.assertEqual(open_fours, [(4, 10), (8, 6)])
        line = board.describe_threats(state, board.BLACK)[0]
        self.assertIn("(4, 10)、(8, 6) 就成活四", line)
        self.assertIn("挡不住", line)

    def test_calls_a_blocked_three_only_a_four_maker(self) -> None:
        state = empty_board()
        for col in range(8, 11):
            state[7][col] = board.BLACK
        state[7][7] = board.WHITE
        open_fours, fours = board.four_makers(state, board.BLACK)
        self.assertEqual(open_fours, [])
        self.assertEqual(fours, [(7, 11), (7, 12)])

    def test_finds_the_gap_of_a_broken_three(self) -> None:
        state = empty_board()
        for col in (8, 9, 11):
            state[7][col] = board.BLACK
        open_fours, _ = board.four_makers(state, board.BLACK)
        self.assertEqual(open_fours, [(7, 10)])

    def test_reports_nothing_for_a_loose_position(self) -> None:
        state = empty_board()
        state[5][5] = board.BLACK
        self.assertEqual(board.describe_threats(state, board.BLACK), [])

    def test_parses_a_json_move(self) -> None:
        self.assertEqual(board.parse_move('好的：{"move": {"row": 7, "col": 7}}'), (7, 7))
        self.assertEqual(board.parse_move('{"move": {"row": 0, "col": 14}}'), (0, 14))
        self.assertIsNone(board.parse_move('{"move": {"row": 15, "col": 0}}'))
        self.assertIsNone(board.parse_move("我不知道"))

    def test_reads_row_and_col_as_zero_based_numbers(self) -> None:
        self.assertEqual(board.parse_move('{"row": 5, "col": 13}'), (5, 13))
        self.assertEqual(board.parse_move('{"row": 0, "col": 0}'), (0, 0))
        self.assertIsNone(board.parse_move('{"row": 16, "col": 16}'))

    def test_does_not_read_a_coordinate_out_of_prose(self) -> None:
        self.assertIsNone(board.parse_move("对手在 H8 有子，我下 I9"))

    def test_finds_an_empty_cell(self) -> None:
        state = empty_board()
        state[7][7] = board.BLACK
        cell = board.fallback_cell(state)
        self.assertIsNotNone(cell)
        self.assertNotEqual(cell, (7, 7))
        self.assertEqual(board_state_value(state, cell), 0)

    def test_reports_a_full_board(self) -> None:
        state = [[1 if (row + col) % 2 == 0 else 2 for col in range(15)] for row in range(15)]
        self.assertIsNone(board.fallback_cell(state))


class EngineTests(unittest.TestCase):
    def test_takes_the_move_of_the_model(self) -> None:
        client = FakeClient(['{"move": {"row": 7, "col": 7}}'])
        engine = LLMEngine(client)
        state = empty_board()
        move = engine.choose_move(state, [], board.BLACK)
        self.assertEqual(move, (7, 7))
        self.assertEqual(len(client.calls), 1)
        system = client.calls[0][0]
        self.assertEqual(system.role, "system")
        self.assertIn("棋盘是 15 行 × 15 列", client.calls[0][1].content)

    def test_the_prompt_tells_the_size_the_stones_and_the_last_move(self) -> None:
        state = empty_board()
        state[7][7] = board.BLACK
        state[8][8] = board.WHITE
        client = FakeClient(['{"move": {"row": 0, "col": 0}}'])
        engine = LLMEngine(client)
        engine.choose_move(
            state,
            [{"row": 7, "col": 7, "number": 1, "player": board.BLACK}],
            board.WHITE,
            {"row": 7, "col": 7, "player": board.BLACK},
        )
        prompt = client.calls[0][1].content
        self.assertIn("15 行 × 15 列", prompt)
        self.assertIn("我方（白棋）的棋子坐标：(8, 8)", prompt)
        self.assertIn("对方（黑棋）的棋子坐标：(7, 7)", prompt)
        self.assertIn("最后一手是黑棋，下在 (7, 7)", prompt)
        self.assertNotIn("行\\列", prompt)
        self.assertNotIn("必须挡", prompt)

    def test_the_prompt_lists_the_threats_of_both_sides(self) -> None:
        state = empty_board()
        for row, col in ((7, 7), (6, 8), (5, 9), (4, 10)):
            state[row][col] = board.BLACK
        state[7][8] = board.WHITE
        state[7][9] = board.WHITE
        state[7][10] = board.WHITE
        client = FakeClient(['{"move": {"row": 8, "col": 6}}'])
        engine = LLMEngine(client)
        engine.choose_move(state, [], board.WHITE, {"row": 4, "col": 10, "player": board.BLACK})
        prompt = client.calls[0][1].content
        self.assertIn(
            "- 黑棋有 2 个成五点 (3, 11)、(8, 6)，落一子就成五，堵一个也挡不住。",
            prompt,
        )
        self.assertIn("白棋下在 (7, 11)、(7, 12) 就成冲四", prompt)

    def test_the_prompt_says_so_when_a_side_has_no_threat(self) -> None:
        client = FakeClient(['{"move": {"row": 7, "col": 7}}'])
        engine = LLMEngine(client)
        engine.choose_move(empty_board(), [], board.BLACK)
        prompt = client.calls[0][1].content
        self.assertIn("- 没有成五、活四或冲四的落点。", prompt)

    def test_asks_again_when_the_cell_is_taken(self) -> None:
        state = empty_board()
        state[3][3] = board.WHITE
        client = FakeClient(
            ['{"move": {"row": 3, "col": 3}}', '{"move": {"row": 4, "col": 4}}']
        )
        engine = LLMEngine(client, max_attempts=2)
        self.assertEqual(engine.choose_move(state, [], board.BLACK), (4, 4))
        self.assertEqual(len(client.calls), 2)
        problem = client.calls[1][-1].content
        self.assertIn("(3, 3)", problem)
        self.assertIn("已有棋子", problem)

    def test_any_empty_cell_is_accepted(self) -> None:
        state = empty_board()
        state[7][7] = board.BLACK
        client = FakeClient(['{"move": {"row": 5, "col": 1}}'])
        engine = LLMEngine(client, max_attempts=2)
        move = engine.choose_move(state, [], board.WHITE)
        self.assertEqual(move, (5, 1))
        self.assertEqual(len(client.calls), 1)

    def test_plays_a_legal_cell_when_the_model_never_complies(self) -> None:
        state = empty_board()
        state[7][7] = board.BLACK
        client = FakeClient(
            ['{"move": {"row": 7, "col": 7}}', '{"move": {"row": 7, "col": 7}}']
        )
        engine = LLMEngine(client, max_attempts=2)
        move = engine.choose_move(state, [], board.WHITE)
        self.assertNotEqual(move, (7, 7))
        self.assertTrue(board.is_empty(state, *move))

    def test_plays_a_legal_cell_when_the_answer_cannot_be_read(self) -> None:
        client = FakeClient(["不知道", "还是不知道"])
        engine = LLMEngine(client, max_attempts=2)
        move = engine.choose_move(empty_board(), [], board.BLACK)
        self.assertTrue(board.is_empty(empty_board(), *move))

    def test_reports_a_full_board(self) -> None:
        state = [[1 if (row + col) % 2 == 0 else 2 for col in range(15)] for row in range(15)]
        client = FakeClient([])
        engine = LLMEngine(client, max_attempts=2)
        with self.assertRaises(EngineError):
            engine.choose_move(state, [], board.BLACK)


class OpenAIClientTests(unittest.TestCase):
    def test_reads_the_content_of_the_first_choice(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        body = json.dumps({"choices": [{"message": {"content": " hi "}}]}).encode("utf-8")
        answer = client._content_of(body)
        self.assertEqual(answer.text, "hi")
        self.assertFalse(answer.from_reasoning)

    def test_falls_back_to_the_reasoning_of_the_model(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        body = json.dumps(
            {"choices": [{"message": {"content": "", "reasoning_content": "思考"}}]}
        ).encode("utf-8")
        answer = client._content_of(body)
        self.assertEqual(answer.text, "思考")
        self.assertTrue(answer.from_reasoning)

    def test_reports_an_empty_answer_that_ran_out_of_tokens(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        body = json.dumps(
            {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        ).encode("utf-8")
        with self.assertRaises(APIError) as caught:
            client._content_of(body)
        self.assertIn("OPENAI_MAX_TOKENS", str(caught.exception))

    def test_reports_an_answer_without_choices(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        with self.assertRaises(APIError):
            client._content_of(b'{"choices": []}')


class RuntimeTokenTests(unittest.TestCase):
    def test_reads_the_token_of_the_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.runtime.json"
            path.write_text(
                json.dumps(
                    {
                        "engines": [
                            {"engine_id": "classical", "token": "c-token"},
                            {"engine_id": "llm", "token": "l-token"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(read_runtime_token(path, "llm"), "l-token")
            self.assertEqual(read_runtime_token(path, "classical"), "c-token")
            self.assertEqual(read_runtime_token(path, "other"), "")

    def test_reports_a_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            self.assertEqual(read_runtime_token(path, "llm"), "")


class SettingsTokenTests(unittest.TestCase):
    def test_takes_the_token_from_the_environment(self) -> None:
        settings = Settings.from_env(
            {
                "MYGO_ENGINE_TOKEN": "token",
                "OPENAI_API_KEY": "key",
                "OPENAI_DISABLE_THINKING": "1",
            }
        )
        self.assertEqual(settings.engine_token, "token")
        self.assertEqual(settings.engine_id, "llm")
        self.assertEqual(settings.worker_id, "llm-worker-01")
        self.assertTrue(settings.disable_thinking)

    def test_takes_the_token_from_the_runtime_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.runtime.json"
            path.write_text(
                json.dumps({"engines": [{"engine_id": "llm", "token": "file-token"}]}),
                encoding="utf-8",
            )
            settings = Settings.from_env(
                {
                    "MYGO_RUNTIME_CONFIG": str(path),
                    "OPENAI_API_KEY": "key",
                }
            )
            self.assertEqual(settings.engine_token, "file-token")

    def test_reports_a_token_it_cannot_find(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.runtime.json"
            path.write_text(json.dumps({"engines": []}), encoding="utf-8")
            with self.assertRaises(ConfigError) as caught:
                Settings.from_env(
                    {
                        "MYGO_RUNTIME_CONFIG": str(path),
                        "OPENAI_API_KEY": "key",
                    }
                )
            self.assertIn("MYGO_ENGINE_TOKEN", str(caught.exception))


class EngineReasoningTests(unittest.TestCase):
    def test_takes_the_json_out_of_the_reasoning(self) -> None:
        client = FakeClient([('先看看棋盘 {"move": {"row": 7, "col": 7}}', True)])
        engine = LLMEngine(client)
        self.assertEqual(engine.choose_move(empty_board(), [], board.BLACK), (7, 7))

    def test_does_not_read_a_coordinate_from_the_reasoning(self) -> None:
        client = FakeClient([("O 在 H8，下一步可能下在 I9", True)])
        engine = LLMEngine(client, max_attempts=1)
        move = engine.choose_move(empty_board(), [], board.BLACK)
        self.assertEqual(move, (7, 7))


if __name__ == "__main__":
    unittest.main()
