from __future__ import annotations

import json
import urllib.error
import urllib.request


class APIError(RuntimeError):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)

        self.status = status


class ChatMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Answer:
    def __init__(self, text: str, from_reasoning: bool = False) -> None:
        self.text = text
        self.from_reasoning = from_reasoning


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        disable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.disable_thinking = disable_thinking

        self._thinking_known_unsupported = False

    def complete(self, messages: list[ChatMessage]) -> Answer:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking and not self._thinking_known_unsupported:
            payload["thinking"] = {"type": "disabled"}
        try:
            body = self._post(payload)
        except APIError as error:
            if error.status != 400 or "thinking" not in payload:
                raise


            self._thinking_known_unsupported = True
            del payload["thinking"]
            body = self._post(payload)
        return self._content_of(body)

    def _post(self, payload: dict[str, object]) -> bytes:
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
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise APIError(f"模型服务返回 {error.code}: {detail}", error.code) from error
        except urllib.error.URLError as error:
            raise APIError(f"无法连接模型服务: {error.reason}") from error

    def _content_of(self, body: bytes) -> Answer:
        try:
            payload = json.loads(body)
        except ValueError as error:
            raise APIError("模型服务的回答不是合法 JSON") from error
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise APIError("模型服务的回答没有 choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise APIError("模型服务的回答没有 message")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return Answer(content.strip())

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return Answer(reasoning.strip(), from_reasoning=True)

        finish = choice.get("finish_reason")
        if finish == "length":
            raise APIError(
                "模型服务的回答是空的：token 预算在思考阶段就用完了，"
                "请调大 OPENAI_MAX_TOKENS，或用 OPENAI_DISABLE_THINKING=1 关闭思考"
            )
        raise APIError("模型服务的回答是空的")
