from unittest.mock import MagicMock

import pytest

from recall.session import Event, Session
from recall.storage import save


def _anthropic_session(*texts: str) -> Session:
    s = Session(provider="anthropic", model="claude-haiku-4-5-20251001")
    for i, text in enumerate(texts):
        s.add_event(Event(seq=i * 2 + 1, type="request", timestamp="t", payload={"messages": []}))
        s.add_event(Event(
            seq=i * 2 + 2, type="response", timestamp="t", duration_ms=100,
            payload={
                "id": f"msg_{i}", "type": "message", "role": "assistant",
                "model": "claude-haiku-4-5-20251001", "stop_reason": "end_turn",
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
# Anthropic replay
# ---------------------------------------------------------------------------

def test_anthropic_replay_returns_recorded_response(tmp_path):
    path = save(_anthropic_session("Hello there!"), base=tmp_path)
    from recall import replay
    client = MagicMock()
    r = replay(client, path).messages.create(model="m", messages=[])
    assert r.content[0].text == "Hello there!"


def test_anthropic_replay_multiple_calls(tmp_path):
    path = save(_anthropic_session("first", "second", "third"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path)
    assert replayed.messages.create(model="m", messages=[]).content[0].text == "first"
    assert replayed.messages.create(model="m", messages=[]).content[0].text == "second"
    assert replayed.messages.create(model="m", messages=[]).content[0].text == "third"


def test_anthropic_replay_exhausted_raises(tmp_path):
    path = save(_anthropic_session("one"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path)
    replayed.messages.create(model="m", messages=[])
    with pytest.raises(IndexError, match="Replay exhausted"):
        replayed.messages.create(model="m", messages=[])


def test_anthropic_replay_with_patch(tmp_path):
    path = save(_anthropic_session("original"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path, patches={0: {"content": [{"type": "text", "text": "patched", "citations": None}]}})
    r = replayed.messages.create(model="m", messages=[])
    assert r.content[0].text == "patched"


def test_anthropic_stream_replay(tmp_path):
    path = save(_anthropic_session("streamed text"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path)
    with replayed.messages.stream(model="m", messages=[]) as stream:
        texts = list(stream.text_stream)
        final = stream.get_final_message()
    assert "streamed text" in texts
    assert final.content[0].text == "streamed text"


# ---------------------------------------------------------------------------
# OpenAI replay
# ---------------------------------------------------------------------------

def test_openai_replay_returns_recorded_response(tmp_path):
    path = save(_openai_session("Hi from OpenAI!"), base=tmp_path)
    from recall import replay
    r = replay(MagicMock(), path).chat.completions.create(model="gpt-4o", messages=[])
    assert r.choices[0].message.content == "Hi from OpenAI!"


def test_openai_replay_multiple_calls(tmp_path):
    path = save(_openai_session("a", "b"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path)
    r1 = replayed.chat.completions.create(model="gpt-4o", messages=[])
    r2 = replayed.chat.completions.create(model="gpt-4o", messages=[])
    assert r1.choices[0].message.content == "a"
    assert r2.choices[0].message.content == "b"


def test_openai_replay_with_patch(tmp_path):
    path = save(_openai_session("original"), base=tmp_path)
    from recall import replay
    replayed = replay(MagicMock(), path, patches={0: {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "patched"}, "finish_reason": "stop"}]
    }})
    r = replayed.chat.completions.create(model="gpt-4o", messages=[])
    assert r.choices[0].message.content == "patched"
