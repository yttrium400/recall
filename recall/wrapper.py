from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .providers.anthropic import normalize_request, normalize_response
from .session import Event, Session
from .storage import save


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _RecordedMessages:
    def __init__(self, original_messages: Any, session: Session, save_path: Any) -> None:
        self._original = original_messages
        self._session = session
        self._save_path = save_path

    def create(self, **kwargs: Any) -> Any:
        req_event = Event(
            seq=self._session.next_seq(),
            type="request",
            timestamp=_now(),
            payload=normalize_request(kwargs),
        )
        self._session.add_event(req_event)

        model = kwargs.get("model", "")
        if model and not self._session.model:
            self._session.model = model

        t0 = time.monotonic()
        try:
            response = self._original.create(**kwargs)
        except Exception as exc:
            err_event = Event(
                seq=self._session.next_seq(),
                type="error",
                timestamp=_now(),
                payload={"error": str(exc), "type": type(exc).__name__},
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            self._session.add_event(err_event)
            save(self._session)
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)
        resp_event = Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload=normalize_response(response),
            duration_ms=duration_ms,
        )
        self._session.add_event(resp_event)
        save(self._session)
        return response


class RecordedClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._session = Session(provider="anthropic")
        self.messages = _RecordedMessages(client.messages, self._session, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def record(client: Any) -> RecordedClient:
    return RecordedClient(client)
