from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..session import Event, Session
from ..storage import save


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_serializable(obj: Any) -> Any:
    """Recursively convert Anthropic SDK objects to plain dicts/lists for JSON storage."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    return obj


def normalize_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _to_serializable(kwargs)


def normalize_response(response: Any) -> dict[str, Any]:
    try:
        return response.model_dump()
    except AttributeError:
        return {"raw": str(response)}


def reconstruct_message(payload: dict[str, Any]) -> Any:
    try:
        from anthropic.types import Message
        return Message.model_validate(payload)
    except ImportError as exc:
        raise ImportError("anthropic package required: pip install anthropic") from exc


# Approximate pricing per million tokens, mid-2025
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_rate, out_rate) in _COST_TABLE.items():
        if model.startswith(prefix):
            return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return None


# ---------------------------------------------------------------------------
# Fake stream used during replay
# ---------------------------------------------------------------------------

class _FakeStream:
    """Mimics anthropic MessageStream enough for user code to work during replay."""

    def __init__(self, message: Any) -> None:
        self._message = message
        self.text_stream = self._iter_text()

    def _iter_text(self) -> Iterator[str]:
        for block in self._message.content:
            if hasattr(block, "type") and block.type == "text":
                yield block.text

    def get_final_message(self) -> Any:
        return self._message

    def __iter__(self) -> Iterator[Any]:
        yield self._message


class _FakeStreamManager:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> _FakeStream:
        return _FakeStream(self._message)

    def __exit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Streaming context manager wrapper used during recording
# ---------------------------------------------------------------------------

class _RecordingStreamManager:
    """Wraps the real Anthropic stream context manager; records on exit."""

    def __init__(self, original_cm: Any, session: Session) -> None:
        self._cm = original_cm
        self._session = session
        self._t0 = time.monotonic()
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._cm.__enter__()
        return self._stream

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = self._cm.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._stream is not None:
            try:
                message = self._stream.get_final_message()
                duration_ms = int((time.monotonic() - self._t0) * 1000)
                resp_event = Event(
                    seq=self._session.next_seq(),
                    type="response",
                    timestamp=_now(),
                    payload=normalize_response(message),
                    duration_ms=duration_ms,
                )
                self._session.add_event(resp_event)
                save(self._session)
            except Exception:
                pass
        return result


# ---------------------------------------------------------------------------
# Recorded messages wrapper
# ---------------------------------------------------------------------------

class AnthropicRecordedMessages:
    def __init__(self, original: Any, session: Session) -> None:
        self._original = original
        self._session = session

    def create(self, **kwargs: Any) -> Any:
        req_event = Event(
            seq=self._session.next_seq(),
            type="request",
            timestamp=_now(),
            payload=normalize_request(kwargs),
        )
        self._session.add_event(req_event)
        if not self._session.model:
            self._session.model = kwargs.get("model", "")

        t0 = time.monotonic()
        try:
            response = self._original.create(**kwargs)
        except Exception as exc:
            self._session.add_event(Event(
                seq=self._session.next_seq(),
                type="error",
                timestamp=_now(),
                payload={"error": str(exc), "type": type(exc).__name__},
                duration_ms=int((time.monotonic() - t0) * 1000),
            ))
            save(self._session)
            raise

        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload=normalize_response(response),
            duration_ms=int((time.monotonic() - t0) * 1000),
        ))
        save(self._session)
        return response

    def stream(self, **kwargs: Any) -> _RecordingStreamManager:
        req_event = Event(
            seq=self._session.next_seq(),
            type="request",
            timestamp=_now(),
            payload=normalize_request(kwargs),
        )
        self._session.add_event(req_event)
        if not self._session.model:
            self._session.model = kwargs.get("model", "")
        return _RecordingStreamManager(self._original.stream(**kwargs), self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Replay messages wrapper
# ---------------------------------------------------------------------------

class AnthropicReplayMessages:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self._responses = [e for e in session.events if e.type == "response"]
        self._patches = patches
        self._call_index = 0

    def _next_message(self) -> Any:
        idx = self._call_index
        self._call_index += 1
        if idx >= len(self._responses):
            raise IndexError(
                f"Replay exhausted: session has {len(self._responses)} recorded response(s), "
                f"but this is call #{idx + 1}"
            )
        payload = dict(self._responses[idx].payload)
        if idx in self._patches:
            payload.update(self._patches[idx])
        return reconstruct_message(payload)

    def create(self, **_kwargs: Any) -> Any:
        return self._next_message()

    def stream(self, **_kwargs: Any) -> _FakeStreamManager:
        return _FakeStreamManager(self._next_message())
