from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Iterator

from ..session import Event, Session
from ..storage import save


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_serializable(obj: Any) -> Any:
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


def reconstruct_completion(payload: dict[str, Any]) -> Any:
    try:
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(payload)
    except ImportError as exc:
        raise ImportError("openai package required: pip install openai") from exc


# Approximate pricing per million tokens, mid-2025
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "o3": (10.00, 40.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_rate, out_rate) in _COST_TABLE.items():
        if model.startswith(prefix):
            return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return None


# ---------------------------------------------------------------------------
# Fake stream for replay
# ---------------------------------------------------------------------------

class _FakeOpenAIStream:
    """Mimics openai Stream[ChatCompletionChunk] for replay."""

    def __init__(self, completion: Any) -> None:
        self._completion = completion

    def __iter__(self) -> Iterator[Any]:
        try:
            from openai.types.chat import ChatCompletionChunk
            from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
        except ImportError:
            return

        content = ""
        choices = self._completion.choices
        if choices:
            msg = choices[0].message
            content = msg.content or ""

        yield ChatCompletionChunk.model_validate({
            "id": self._completion.id,
            "object": "chat.completion.chunk",
            "created": self._completion.created,
            "model": self._completion.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        })
        yield ChatCompletionChunk.model_validate({
            "id": self._completion.id,
            "object": "chat.completion.chunk",
            "created": self._completion.created,
            "model": self._completion.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })

    def __enter__(self) -> "_FakeOpenAIStream":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Streaming wrapper for recording
# ---------------------------------------------------------------------------

class _RecordingOpenAIStream:
    """Wraps openai Stream[ChatCompletionChunk]; records aggregated response on exhaustion."""

    def __init__(self, original_stream: Any, session: Session, t0: float) -> None:
        self._stream = original_stream
        self._session = session
        self._t0 = t0
        self._chunks: list[dict[str, Any]] = []
        self._recorded = False

    def __iter__(self) -> Iterator[Any]:
        for chunk in self._stream:
            self._chunks.append(chunk.model_dump())
            yield chunk
        self._record()

    def __enter__(self) -> "_RecordingOpenAIStream":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        result = None
        if hasattr(self._stream, "__exit__"):
            result = self._stream.__exit__(*args)
        if not self._recorded:
            self._record()
        return result

    def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload={"streaming": True, "chunks": self._chunks},
            duration_ms=duration_ms,
        ))
        save(self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Recorded completions wrapper
# ---------------------------------------------------------------------------

class OpenAIRecordedCompletions:
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
        streaming = kwargs.get("stream", False)

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

        if streaming:
            return _RecordingOpenAIStream(response, self._session, t0)

        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload=normalize_response(response),
            duration_ms=int((time.monotonic() - t0) * 1000),
        ))
        save(self._session)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class OpenAIRecordedChat:
    def __init__(self, original_chat: Any, session: Session) -> None:
        self.completions = OpenAIRecordedCompletions(original_chat.completions, session)
        self._original = original_chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Replay completions wrapper
# ---------------------------------------------------------------------------

class OpenAIReplayCompletions:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self._responses = [e for e in session.events if e.type == "response"]
        self._patches = patches
        self._call_index = 0

    def _next_completion(self) -> Any:
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
        if payload.get("streaming"):
            return payload
        return reconstruct_completion(payload)

    def create(self, **kwargs: Any) -> Any:
        completion = self._next_completion()
        if isinstance(completion, dict) and completion.get("streaming"):
            if kwargs.get("stream"):
                chunks_raw = completion.get("chunks", [])
                try:
                    from openai.types.chat import ChatCompletionChunk
                    chunks = [ChatCompletionChunk.model_validate(c) for c in chunks_raw]
                except ImportError:
                    chunks = []
                return iter(chunks)
            return completion
        return completion


class OpenAIReplayChat:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self.completions = OpenAIReplayCompletions(session, patches)


# ---------------------------------------------------------------------------
# Async streaming wrapper for recording
# ---------------------------------------------------------------------------

class _AsyncRecordingOpenAIStream:
    """Wraps openai AsyncStream[ChatCompletionChunk]; records aggregated response on exhaustion."""

    def __init__(self, original_stream: Any, session: Session, t0: float) -> None:
        self._stream = original_stream
        self._session = session
        self._t0 = t0
        self._chunks: list[dict[str, Any]] = []
        self._recorded = False

    def __aiter__(self) -> "_AsyncRecordingOpenAIStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk.model_dump())
            return chunk
        except StopAsyncIteration:
            await self._record()
            raise

    async def __aenter__(self) -> "_AsyncRecordingOpenAIStream":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        result = None
        if hasattr(self._stream, "__aexit__"):
            result = await self._stream.__aexit__(*args)
        if not self._recorded:
            await self._record()
        return result

    async def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload={"streaming": True, "chunks": self._chunks},
            duration_ms=duration_ms,
        ))
        await asyncio.to_thread(save, self._session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Async fake stream for replay
# ---------------------------------------------------------------------------

class _AsyncFakeOpenAIStream:
    """Mimics openai AsyncStream[ChatCompletionChunk] for replay."""

    def __init__(self, chunks: list[Any]) -> None:
        self._iter = iter(chunks)

    def __aiter__(self) -> "_AsyncFakeOpenAIStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self) -> "_AsyncFakeOpenAIStream":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Async recorded completions wrapper
# ---------------------------------------------------------------------------

class AsyncOpenAIRecordedCompletions:
    def __init__(self, original: Any, session: Session) -> None:
        self._original = original
        self._session = session

    async def create(self, **kwargs: Any) -> Any:
        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="request",
            timestamp=_now(),
            payload=normalize_request(kwargs),
        ))
        if not self._session.model:
            self._session.model = kwargs.get("model", "")

        t0 = time.monotonic()
        streaming = kwargs.get("stream", False)

        try:
            response = await self._original.create(**kwargs)
        except Exception as exc:
            self._session.add_event(Event(
                seq=self._session.next_seq(),
                type="error",
                timestamp=_now(),
                payload={"error": str(exc), "type": type(exc).__name__},
                duration_ms=int((time.monotonic() - t0) * 1000),
            ))
            await asyncio.to_thread(save, self._session)
            raise

        if streaming:
            return _AsyncRecordingOpenAIStream(response, self._session, t0)

        self._session.add_event(Event(
            seq=self._session.next_seq(),
            type="response",
            timestamp=_now(),
            payload=normalize_response(response),
            duration_ms=int((time.monotonic() - t0) * 1000),
        ))
        await asyncio.to_thread(save, self._session)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncOpenAIRecordedChat:
    def __init__(self, original_chat: Any, session: Session) -> None:
        self.completions = AsyncOpenAIRecordedCompletions(original_chat.completions, session)
        self._original = original_chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Async replay completions wrapper
# ---------------------------------------------------------------------------

class AsyncOpenAIReplayCompletions:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self._responses = [e for e in session.events if e.type == "response"]
        self._patches = patches
        self._call_index = 0

    def _next_completion(self) -> Any:
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
        if payload.get("streaming"):
            return payload
        return reconstruct_completion(payload)

    async def create(self, **kwargs: Any) -> Any:
        completion = self._next_completion()
        if isinstance(completion, dict) and completion.get("streaming"):
            if kwargs.get("stream"):
                chunks_raw = completion.get("chunks", [])
                try:
                    from openai.types.chat import ChatCompletionChunk
                    chunks = [ChatCompletionChunk.model_validate(c) for c in chunks_raw]
                except ImportError:
                    chunks = []
                return _AsyncFakeOpenAIStream(chunks)
            return completion
        return completion


class AsyncOpenAIReplayChat:
    def __init__(self, session: Session, patches: dict[int, dict[str, Any]]) -> None:
        self.completions = AsyncOpenAIReplayCompletions(session, patches)
