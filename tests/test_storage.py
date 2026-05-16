from recall.session import Event, Session
from recall.storage import list_sessions, load, save


def test_save_and_load(tmp_path):
    s = Session(provider="anthropic", model="claude-sonnet-4-6")
    s.add_event(Event(seq=1, type="request", timestamp="2026-05-16T10:00:00Z", payload={"x": 1}))
    path = save(s, base=tmp_path)
    assert path.exists()
    restored = load(path)
    assert restored.id == s.id
    assert restored.model == "claude-sonnet-4-6"
    assert restored.events[0].payload == {"x": 1}


def test_list_sessions(tmp_path):
    assert list_sessions(tmp_path) == []
    s1 = Session()
    s2 = Session()
    save(s1, base=tmp_path)
    save(s2, base=tmp_path)
    assert len(list_sessions(tmp_path)) == 2


def test_save_creates_directory(tmp_path):
    target = tmp_path / "nested"
    s = Session()
    path = save(s, base=target)
    assert path.exists()
