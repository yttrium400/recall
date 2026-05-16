from __future__ import annotations

from pathlib import Path
from typing import Any

from .providers.anthropic import reconstruct_message
from .session import Session
from .storage import load


class _ReplayMessages:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self._responses = [
            e for e in session.events if e.type == "response"
        ]
        self._patches = patches
        self._call_index = 0

    def create(self, **kwargs: Any) -> Any:
        idx = self._call_index
        self._call_index += 1

        if idx >= len(self._responses):
            raise IndexError(
                f"Replay exhausted: session has {len(self._responses)} recorded response(s), "
                f"but this is call #{idx + 1}"
            )

        payload = dict(self._responses[idx].payload)

        if idx in self._patches:
            patch = self._patches[idx]
            payload.update(patch)

        return reconstruct_message(payload)


class ReplayClient:
    def __init__(self, client: Any, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self._client = client
        self._session = session
        self.messages = _ReplayMessages(session, patches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def replay(
    client: Any,
    session: str | Path,
    patches: dict[int, dict[str, Any]] | None = None,
) -> ReplayClient:
    """Return a client that replays recorded responses instead of making API calls.

    patches maps call index (0-based) to a dict that overrides fields of that response.
    """
    path = Path(session)
    loaded = load(path)
    return ReplayClient(client, loaded, patches or {})
