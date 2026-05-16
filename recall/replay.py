from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import load


class AnthropicReplayClient:
    def __init__(self, client: Any, session: Any, patches: dict[int, dict[str, Any]]) -> None:
        self._client = client
        self._session = session
        from .providers.anthropic import AnthropicReplayMessages
        self.messages = AnthropicReplayMessages(session, patches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class OpenAIReplayClient:
    def __init__(self, client: Any, session: Any, patches: dict[int, dict[str, Any]]) -> None:
        self._client = client
        self._session = session
        from .providers.openai import OpenAIReplayChat
        self.chat = OpenAIReplayChat(session, patches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def replay(
    client: Any,
    session: str | Path,
    patches: dict[int, dict[str, Any]] | None = None,
) -> AnthropicReplayClient | OpenAIReplayClient:
    """Return a client that replays recorded responses instead of making API calls.

    patches maps call index (0-based) to a dict merged into that response payload.
    Subsequent calls after a patched index still return their recorded responses -
    useful for testing error handling at a specific step.
    """
    loaded = load(Path(session))
    p = patches or {}
    if loaded.provider == "openai":
        return OpenAIReplayClient(client, loaded, p)
    return AnthropicReplayClient(client, loaded, p)
