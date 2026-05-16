from unittest.mock import MagicMock, patch

import pytest

from recall.session import Event, Session
from recall.replay import ReplayClient, _ReplayMessages


def _session_with_responses(*texts: str) -> Session:
    s = Session(provider="anthropic", model="claude-haiku-4-5-20251001")
    for i, text in enumerate(texts):
        s.add_event(Event(seq=i * 2 + 1, type="request", timestamp="t", payload={"messages": []}))
        s.add_event(Event(
            seq=i * 2 + 2,
            type="response",
            timestamp="t",
            payload={
                "id": f"msg_{i}",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": text, "citations": None}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "stop_details": None,
                "container": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                    "server_tool_use": None,
                    "service_tier": "standard",
                    "inference_geo": "not_available",
                },
            },
            duration_ms=100,
        ))
    return s


def test_replay_returns_recorded_response(tmp_path):
    session = _session_with_responses("Hello there!")
    from recall.storage import save
    path = save(session, base=tmp_path)

    client = MagicMock()
    from recall.replay import replay
    replayed = replay(client, path)
    response = replayed.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    assert response.content[0].text == "Hello there!"


def test_replay_exhausted_raises(tmp_path):
    session = _session_with_responses("one")
    from recall.storage import save
    path = save(session, base=tmp_path)

    client = MagicMock()
    from recall.replay import replay
    replayed = replay(client, path)
    replayed.messages.create(model="m", messages=[])
    with pytest.raises(IndexError, match="Replay exhausted"):
        replayed.messages.create(model="m", messages=[])


def test_replay_with_patch(tmp_path):
    session = _session_with_responses("original answer")
    from recall.storage import save
    path = save(session, base=tmp_path)

    client = MagicMock()
    from recall.replay import replay
    replayed = replay(client, path, patches={0: {"content": [{"type": "text", "text": "patched answer", "citations": None}]}})
    response = replayed.messages.create(model="m", messages=[])
    assert response.content[0].text == "patched answer"


def test_replay_multiple_calls(tmp_path):
    session = _session_with_responses("first", "second", "third")
    from recall.storage import save
    path = save(session, base=tmp_path)

    client = MagicMock()
    from recall.replay import replay
    replayed = replay(client, path)
    r1 = replayed.messages.create(model="m", messages=[])
    r2 = replayed.messages.create(model="m", messages=[])
    r3 = replayed.messages.create(model="m", messages=[])
    assert r1.content[0].text == "first"
    assert r2.content[0].text == "second"
    assert r3.content[0].text == "third"
