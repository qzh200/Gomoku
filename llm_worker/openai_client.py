"""A client of the OpenAI chat completions protocol.

The client speaks the OpenAI protocol, so it works with OpenAI itself and
with every service that copies it: DeepSeek, Moonshot, Ollama, vLLM, LM
Studio and the like. The endpoint is chosen with the base URL.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class APIError(RuntimeError):
    """The model service refused the request or answered with an error."""


class ChatMessage:
    """One message of a conversation."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_payload(self) -> dict[str, str]:
        """Return the wire form of the message."""
        return {"role": self.role, "content": self.content}


class OpenAIClient:
    """Talks to an OpenAI compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 200,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, messages: list[ChatMessage]) -> str:
        """Ask the model for the next message and return its content."""
        payload = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise APIError(f"模型服务返回 {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise APIError(f"无法连接模型服务: {error.reason}") from error
        return self._content_of(body)

    def _content_of(self, body: bytes) -> str:
        """Read the content of the first choice of the answer."""
        try:
            payload = json.loads(body)
        except ValueError as error:
            raise APIError("模型服务的回答不是合法 JSON") from error
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise APIError("模型服务的回答没有 choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIError("模型服务的回答没有 message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise APIError("模型服务的回答是空的")
        return content.strip()
