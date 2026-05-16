from __future__ import annotations

from pathlib import Path
from typing import Any

from .session import Session
from .storage import save


def _detect_provider(client: Any) -> str:
    module = type(client).__module__
    if module.startswith("anthropic"):
        return "anthropic"
    if module.startswith("openai"):
        return "openai"
    raise ValueError(
        f"Unsupported client type: {type(client).__qualname__}. "
        "Expected an anthropic.Anthropic or openai.OpenAI instance."
    )


class AnthropicRecordedClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._session = Session(provider="anthropic")
        from .providers.anthropic import AnthropicRecordedMessages
        self.messages = AnthropicRecordedMessages(client.messages, self._session)

    @property
    def session_path(self) -> Path:
        return save(self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class OpenAIRecordedClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._session = Session(provider="openai")
        from .providers.openai import OpenAIRecordedChat
        self.chat = OpenAIRecordedChat(client.chat, self._session)

    @property
    def session_path(self) -> Path:
        return save(self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def record(client: Any) -> AnthropicRecordedClient | OpenAIRecordedClient:
    provider = _detect_provider(client)
    if provider == "anthropic":
        return AnthropicRecordedClient(client)
    return OpenAIRecordedClient(client)
