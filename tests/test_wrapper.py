from unittest.mock import MagicMock, patch

import pytest

from recall.session import Session
from recall.wrapper import RecordedClient, record


def _mock_response(model: str = "claude-sonnet-4-6") -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = {"id": "msg_123", "model": model, "content": []}
    return resp


def test_record_returns_recorded_client():
    client = MagicMock()
    wrapped = record(client)
    assert isinstance(wrapped, RecordedClient)


def test_messages_create_records_request_and_response(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    wrapped = RecordedClient(client)

    with patch("recall.wrapper.save") as mock_save:
        wrapped.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
        assert mock_save.called

    events = wrapped._session.events
    assert len(events) == 2
    assert events[0].type == "request"
    assert events[1].type == "response"
    assert events[1].duration_ms is not None


def test_model_extracted_from_request(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("claude-opus-4-7")

    wrapped = RecordedClient(client)
    with patch("recall.wrapper.save"):
        wrapped.messages.create(model="claude-opus-4-7", messages=[])

    assert wrapped._session.model == "claude-opus-4-7"


def test_error_recorded_and_reraised():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("API down")

    wrapped = RecordedClient(client)
    with patch("recall.wrapper.save"):
        with pytest.raises(RuntimeError, match="API down"):
            wrapped.messages.create(model="claude-sonnet-4-6", messages=[])

    events = wrapped._session.events
    assert len(events) == 2
    assert events[1].type == "error"
    assert events[1].payload["error"] == "API down"


def test_passthrough_attribute():
    client = MagicMock()
    client.some_attr = "value"
    wrapped = record(client)
    assert wrapped.some_attr == "value"
