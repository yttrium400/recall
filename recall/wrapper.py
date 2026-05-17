from __future__ import annotations

from pathlib import Path
from typing import Any

from .session import Session
from .storage import save


def _detect_provider(client: Any) -> tuple[str, bool]:
    module = type(client).__module__
    name = type(client).__name__
    is_async = "Async" in name
    if module.startswith("anthropic"):
        return "anthropic", is_async
    if module.startswith("openai"):
        return "openai", is_async
    raise ValueError(
        f"Unsupported client type: {type(client).__qualname__}. "
        "Expected anthropic.Anthropic, anthropic.AsyncAnthropic, openai.OpenAI, or openai.AsyncOpenAI."
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


class AsyncAnthropicRecordedClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._session = Session(provider="anthropic")
        from .providers.anthropic import AsyncAnthropicRecordedMessages
        self.messages = AsyncAnthropicRecordedMessages(client.messages, self._session)

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


class AsyncOpenAIRecordedClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._session = Session(provider="openai")
        from .providers.openai import AsyncOpenAIRecordedChat
        self.chat = AsyncOpenAIRecordedChat(client.chat, self._session)

    @property
    def session_path(self) -> Path:
        return save(self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def record(client: Any) -> AnthropicRecordedClient | AsyncAnthropicRecordedClient | OpenAIRecordedClient | AsyncOpenAIRecordedClient:
    provider, is_async = _detect_provider(client)
    if provider == "anthropic":
        return AsyncAnthropicRecordedClient(client) if is_async else AnthropicRecordedClient(client)
    return AsyncOpenAIRecordedClient(client) if is_async else OpenAIRecordedClient(client)
