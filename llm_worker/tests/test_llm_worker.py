"""Tests of the LLM engine.

The tests run with the standard library only:
    python -m unittest discover -s llm_worker/tests -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_worker import board
from llm_worker.config import ConfigError, Settings, parse_dotenv, read_runtime_token
from llm_worker.engine import EngineError, LLMEngine
from llm_worker.openai_client import APIError, Answer, ChatMessage, OpenAIClient


class FakeClient:
    """A model client that answers with a fixed list of answers.

    An answer given as a tuple marks the text as the reasoning of a model.
    """

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
    """Return an empty 15x15 board."""
    return [[0] * board.BOARD_SIZE for _ in range(board.BOARD_SIZE)]


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
    def test_renders_the_labels_of_the_board(self) -> None:
        state = empty_board()
        state[0][0] = board.BLACK
        state[1][2] = board.WHITE
        text = board.render_board(state)
        self.assertIn("A B C D E F G H I J K L M N O", text)
        self.assertIn("01 X . .", text)
        self.assertIn("02 . . O", text)
        self.assertIn("15", text)

    def test_parses_a_coordinate_of_the_labels(self) -> None:
        self.assertEqual(board.parse_coordinate("H8"), (7, 7))
        self.assertEqual(board.parse_coordinate("h8"), (7, 7))
        self.assertEqual(board.parse_coordinate("落点 A1"), (0, 0))
        self.assertIsNone(board.parse_coordinate("P9"))
        self.assertIsNone(board.parse_coordinate("没有坐标"))

    def test_parses_a_json_move(self) -> None:
        self.assertEqual(board.parse_move('好的：{"move": "H8"}'), (7, 7))
        self.assertEqual(board.parse_move('{"move": "M5"}'), (4, 12))
        self.assertEqual(board.parse_move("H8"), (7, 7))
        self.assertIsNone(board.parse_move("我不知道"))

    def test_reads_numbers_as_the_labels_of_the_board(self) -> None:
        # M5 on the board is the row 5 and the column 13, so it is the cell
        # (4, 12) of the API.
        self.assertEqual(board.parse_move('{"row": 5, "col": 13}'), (4, 12))
        self.assertEqual(board.parse_move('{"row": 1, "col": 1}'), (0, 0))
        # Out of the labels, the numbers are the 0-based indexes of the API.
        self.assertEqual(board.parse_move('{"row": 0, "col": 0}'), (0, 0))
        self.assertIsNone(board.parse_move('{"row": 16, "col": 16}'))

    def test_ignores_the_reasoning_prose(self) -> None:
        self.assertIsNone(board.parse_move_strict("对手在 M8 有子，我下 I9"))
        self.assertEqual(board.parse_move_strict('{"move": "I9"}'), (8, 8))

    def test_offers_the_empty_cells_next_to_the_stones(self) -> None:
        state = empty_board()
        state[7][7] = board.BLACK
        cells = board.candidates(state)
        self.assertNotIn((7, 7), cells)
        self.assertIn((7, 6), cells)
        self.assertIn((8, 8), cells)
        # A far corner is not worth offering.
        self.assertNotIn((0, 0), cells)
        # The cells next to a stone come before the ones further out.
        self.assertIn((7, 6), cells[:8])

    def test_offers_the_center_on_an_empty_board(self) -> None:
        self.assertEqual(board.candidates(empty_board()), [(7, 7)])


class EngineTests(unittest.TestCase):
    def test_takes_the_move_of_the_model(self) -> None:
        client = FakeClient(['{"move": "H8"}'])
        engine = LLMEngine(client)
        state = empty_board()
        move = engine.choose_move(state, [], board.BLACK)
        self.assertEqual(move, (7, 7))
        self.assertEqual(len(client.calls), 1)
        system = client.calls[0][0]
        self.assertEqual(system.role, "system")
        self.assertIn("A B C", client.calls[0][1].content)

    def test_asks_again_when_the_cell_is_taken(self) -> None:
        state = empty_board()
        state[3][3] = board.WHITE
        client = FakeClient(['{"move": "D4"}', '{"move": "E5"}'])
        engine = LLMEngine(client, max_attempts=2)
        self.assertEqual(engine.choose_move(state, [], board.BLACK), (4, 4))
        self.assertEqual(len(client.calls), 2)
        problem = client.calls[1][-1].content
        self.assertIn("D4", problem)
        self.assertIn("已有棋子", problem)

    def test_gives_up_when_no_answer_is_playable(self) -> None:
        client = FakeClient(["不知道", "还是不知道"])
        engine = LLMEngine(client, max_attempts=2)
        with self.assertRaises(EngineError):
            engine.choose_move(empty_board(), [], board.BLACK)


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
        client = FakeClient([('先看看棋盘 {"move": "H8"}', True)])
        engine = LLMEngine(client)
        self.assertEqual(engine.choose_move(empty_board(), [], board.BLACK), (7, 7))

    def test_does_not_read_a_coordinate_from_the_reasoning(self) -> None:
        client = FakeClient([("O 在 H8，下一步可能下在 I9", True)])
        engine = LLMEngine(client, max_attempts=1)
        with self.assertRaises(EngineError):
            engine.choose_move(empty_board(), [], board.BLACK)


if __name__ == "__main__":
    unittest.main()
