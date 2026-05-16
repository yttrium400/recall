import json

from recall.session import Event, Session


def test_event_roundtrip():
    e = Event(seq=1, type="request", timestamp="2026-05-16T10:00:00Z", payload={"model": "x"})
    assert Event.from_dict(e.to_dict()).seq == 1
    assert Event.from_dict(e.to_dict()).payload == {"model": "x"}


def test_event_response_with_duration():
    e = Event(seq=2, type="response", timestamp="2026-05-16T10:00:01Z", payload={}, duration_ms=500)
    d = e.to_dict()
    assert d["duration_ms"] == 500
    assert Event.from_dict(d).duration_ms == 500


def test_session_roundtrip():
    s = Session(provider="anthropic", model="claude-sonnet-4-6")
    e = Event(seq=1, type="request", timestamp="2026-05-16T10:00:00Z", payload={"a": 1})
    s.add_event(e)
    restored = Session.from_dict(s.to_dict())
    assert restored.model == "claude-sonnet-4-6"
    assert len(restored.events) == 1
    assert restored.events[0].payload == {"a": 1}


def test_session_next_seq():
    s = Session()
    assert s.next_seq() == 1
    s.add_event(Event(seq=1, type="request", timestamp="t", payload={}))
    assert s.next_seq() == 2


def test_session_serializes_to_json():
    s = Session(id="abc", provider="anthropic", model="m", started_at="2026-05-16T10:00:00Z")
    d = s.to_dict()
    assert json.dumps(d)  # must be JSON-serializable
    assert d["id"] == "abc"
