"""Tests of the LLM engine.

The tests run with the standard library only:
    python -m unittest discover -s llm_worker/tests -t .
"""

from __future__ import annotations

import json
import unittest

from llm_worker import board
from llm_worker.config import ConfigError, Settings, parse_dotenv
from llm_worker.engine import EngineError, LLMEngine
from llm_worker.openai_client import APIError, ChatMessage, OpenAIClient


class FakeClient:
    """A model client that answers with a fixed list of answers."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        if not self.answers:
            raise APIError("没有更多回答")
        return self.answers.pop(0)


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
        with self.assertRaises(ConfigError):
            Settings.from_env({"OPENAI_API_KEY": "key"})

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
        self.assertEqual(board.parse_move('好的：{"row": 7, "col": 8}'), (7, 8))
        self.assertEqual(board.parse_move("H8"), (7, 7))
        self.assertIsNone(board.parse_move("我不知道"))


class EngineTests(unittest.TestCase):
    def test_takes_the_move_of_the_model(self) -> None:
        client = FakeClient(['{"row": 7, "col": 8}'])
        engine = LLMEngine(client)
        state = empty_board()
        move = engine.choose_move(state, [], board.BLACK)
        self.assertEqual(move, (7, 8))
        self.assertEqual(len(client.calls), 1)
        system = client.calls[0][0]
        self.assertEqual(system.role, "system")
        self.assertIn("A B C", client.calls[0][1].content)

    def test_asks_again_when_the_cell_is_taken(self) -> None:
        state = empty_board()
        state[3][3] = board.WHITE
        client = FakeClient(['{"row": 3, "col": 3}', '{"row": 4, "col": 4}'])
        engine = LLMEngine(client, max_attempts=2)
        self.assertEqual(engine.choose_move(state, [], board.BLACK), (4, 4))
        self.assertEqual(len(client.calls), 2)
        self.assertIn("已有棋子", client.calls[1][-1].content)

    def test_gives_up_when_no_answer_is_playable(self) -> None:
        client = FakeClient(["不知道", "还是不知道"])
        engine = LLMEngine(client, max_attempts=2)
        with self.assertRaises(EngineError):
            engine.choose_move(empty_board(), [], board.BLACK)


class OpenAIClientTests(unittest.TestCase):
    def test_reads_the_content_of_the_first_choice(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        body = json.dumps({"choices": [{"message": {"content": " hi "}}]}).encode("utf-8")
        self.assertEqual(client._content_of(body), "hi")

    def test_reports_an_answer_without_choices(self) -> None:
        client = OpenAIClient("http://127.0.0.1:1/v1", "key", "model")
        with self.assertRaises(APIError):
            client._content_of(b'{"choices": []}')


if __name__ == "__main__":
    unittest.main()
