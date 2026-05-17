"""Tests for async recording and replay (Anthropic + OpenAI)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recall.session import Event, Session
from recall.storage import save
from recall.wrapper import AsyncAnthropicRecordedClient, AsyncOpenAIRecordedClient, record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_anthropic_response(text: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "stop_sequence": None, "stop_details": None, "container": None,
        "content": [{"type": "text", "text": text, "citations": None}],
        "usage": {
            "input_tokens": 10, "output_tokens": 5,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "server_tool_use": None, "service_tier": "standard", "inference_geo": "not_available",
        },
    }
    return resp


def _mock_openai_response(text: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "chatcmpl_1", "object": "chat.completion", "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return resp


def _make_async_anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic"
    client.__class__.__name__ = "AsyncAnthropic"
    return client


def _make_async_openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai"
    client.__class__.__name__ = "AsyncOpenAI"
    return client


def _anthropic_session(*texts: str) -> Session:
    s = Session(provider="anthropic", model="claude-sonnet-4-6")
    for i, text in enumerate(texts):
        s.add_event(Event(seq=i * 2 + 1, type="request", timestamp="t", payload={"messages": []}))
        s.add_event(Event(
            seq=i * 2 + 2, type="response", timestamp="t", duration_ms=100,
            payload={
                "id": f"msg_{i}", "type": "message", "role": "assistant",
                "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
                "stop_sequence": None, "stop_details": None, "container": None,
                "content": [{"type": "text", "text": text, "citations": None}],
                "usage": {
                    "input_tokens": 10, "output_tokens": 5,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                    "server_tool_use": None, "service_tier": "standard", "inference_geo": "not_available",
                },
            },
        ))
    return s


def _openai_session(*texts: str) -> Session:
    s = Session(provider="openai", model="gpt-4o")
    for i, text in enumerate(texts):
        s.add_event(Event(seq=i * 2 + 1, type="request", timestamp="t", payload={"messages": []}))
        s.add_event(Event(
            seq=i * 2 + 2, type="response", timestamp="t", duration_ms=100,
            payload={
                "id": f"chatcmpl_{i}", "object": "chat.completion", "created": 0,
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ))
    return s


# ---------------------------------------------------------------------------
# record() detection
# ---------------------------------------------------------------------------

def test_record_async_anthropic_returns_correct_client():
    client = _make_async_anthropic_client()
    wrapped = record(client)
    assert isinstance(wrapped, AsyncAnthropicRecordedClient)


def test_record_async_openai_returns_correct_client():
    client = _make_async_openai_client()
    wrapped = record(client)
    assert isinstance(wrapped, AsyncOpenAIRecordedClient)


# ---------------------------------------------------------------------------
# Async Anthropic recording
# ---------------------------------------------------------------------------

def test_async_anthropic_create_records_events():
    async def run():
        client = _make_async_anthropic_client()
        client.messages.create = AsyncMock(return_value=_mock_anthropic_response("hi"))
        wrapped = AsyncAnthropicRecordedClient(client)
        with patch("recall.providers.anthropic.save"):
            await wrapped.messages.create(model="claude-sonnet-4-6", messages=[])
        assert len(wrapped._session.events) == 2
        assert wrapped._session.events[0].type == "request"
        assert wrapped._session.events[1].type == "response"

    asyncio.run(run())


def test_async_anthropic_model_extracted():
    async def run():
        client = _make_async_anthropic_client()
        client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        wrapped = AsyncAnthropicRecordedClient(client)
        with patch("recall.providers.anthropic.save"):
            await wrapped.messages.create(model="claude-opus-4-7", messages=[])
        assert wrapped._session.model == "claude-opus-4-7"

    asyncio.run(run())


def test_async_anthropic_error_recorded_and_reraised():
    async def run():
        client = _make_async_anthropic_client()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        wrapped = AsyncAnthropicRecordedClient(client)
        with patch("recall.providers.anthropic.save"):
            with pytest.raises(RuntimeError):
                await wrapped.messages.create(model="m", messages=[])
        assert wrapped._session.events[-1].type == "error"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Async Anthropic replay
# ---------------------------------------------------------------------------

def test_async_anthropic_replay_returns_recorded_response(tmp_path):
    async def run():
        path = save(_anthropic_session("async hello"), base=tmp_path)
        from recall import replay
        client = _make_async_anthropic_client()
        r = await replay(client, path).messages.create(model="m", messages=[])
        assert r.content[0].text == "async hello"

    asyncio.run(run())


def test_async_anthropic_replay_multiple_calls(tmp_path):
    async def run():
        path = save(_anthropic_session("first", "second"), base=tmp_path)
        from recall import replay
        replayed = replay(_make_async_anthropic_client(), path)
        r1 = await replayed.messages.create(model="m", messages=[])
        r2 = await replayed.messages.create(model="m", messages=[])
        assert r1.content[0].text == "first"
        assert r2.content[0].text == "second"

    asyncio.run(run())


def test_async_anthropic_replay_exhausted_raises(tmp_path):
    async def run():
        path = save(_anthropic_session("one"), base=tmp_path)
        from recall import replay
        replayed = replay(_make_async_anthropic_client(), path)
        await replayed.messages.create(model="m", messages=[])
        with pytest.raises(IndexError, match="Replay exhausted"):
            await replayed.messages.create(model="m", messages=[])

    asyncio.run(run())


def test_async_anthropic_replay_with_patch(tmp_path):
    async def run():
        path = save(_anthropic_session("original"), base=tmp_path)
        from recall import replay
        replayed = replay(
            _make_async_anthropic_client(), path,
            patches={0: {"content": [{"type": "text", "text": "patched", "citations": None}]}},
        )
        r = await replayed.messages.create(model="m", messages=[])
        assert r.content[0].text == "patched"

    asyncio.run(run())


def test_async_anthropic_stream_replay(tmp_path):
    async def run():
        path = save(_anthropic_session("streamed text"), base=tmp_path)
        from recall import replay
        replayed = replay(_make_async_anthropic_client(), path)
        async with replayed.messages.stream(model="m", messages=[]) as stream:
            texts = list(stream.text_stream)
            final = stream.get_final_message()
        assert "streamed text" in texts
        assert final.content[0].text == "streamed text"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Async OpenAI recording
# ---------------------------------------------------------------------------

def test_async_openai_create_records_events():
    async def run():
        client = _make_async_openai_client()
        client.chat.completions.create = AsyncMock(return_value=_mock_openai_response())
        wrapped = AsyncOpenAIRecordedClient(client)
        with patch("recall.providers.openai.save"):
            await wrapped.chat.completions.create(model="gpt-4o", messages=[])
        assert len(wrapped._session.events) == 2
        assert wrapped._session.events[0].type == "request"
        assert wrapped._session.events[1].type == "response"

    asyncio.run(run())


def test_async_openai_error_recorded_and_reraised():
    async def run():
        client = _make_async_openai_client()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
        wrapped = AsyncOpenAIRecordedClient(client)
        with patch("recall.providers.openai.save"):
            with pytest.raises(RuntimeError):
                await wrapped.chat.completions.create(model="gpt-4o", messages=[])
        assert wrapped._session.events[-1].type == "error"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Async OpenAI replay
# ---------------------------------------------------------------------------

def test_async_openai_replay_returns_recorded_response(tmp_path):
    async def run():
        path = save(_openai_session("async hi"), base=tmp_path)
        from recall import replay
        r = await replay(_make_async_openai_client(), path).chat.completions.create(model="gpt-4o", messages=[])
        assert r.choices[0].message.content == "async hi"

    asyncio.run(run())


def test_async_openai_replay_multiple_calls(tmp_path):
    async def run():
        path = save(_openai_session("a", "b"), base=tmp_path)
        from recall import replay
        replayed = replay(_make_async_openai_client(), path)
        r1 = await replayed.chat.completions.create(model="gpt-4o", messages=[])
        r2 = await replayed.chat.completions.create(model="gpt-4o", messages=[])
        assert r1.choices[0].message.content == "a"
        assert r2.choices[0].message.content == "b"

    asyncio.run(run())


def test_async_openai_replay_with_patch(tmp_path):
    async def run():
        path = save(_openai_session("original"), base=tmp_path)
        from recall import replay
        replayed = replay(_make_async_openai_client(), path, patches={0: {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "patched"}, "finish_reason": "stop"}]
        }})
        r = await replayed.chat.completions.create(model="gpt-4o", messages=[])
        assert r.choices[0].message.content == "patched"

    asyncio.run(run())
