from unittest.mock import MagicMock, patch

import pytest

from recall.wrapper import AnthropicRecordedClient, OpenAIRecordedClient, record


def _mock_anthropic_response(model: str = "claude-sonnet-4-6") -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = {"id": "msg_123", "model": model, "content": [], "type": "message",
                                     "role": "assistant", "stop_reason": "end_turn", "usage": {}}
    return resp


def _mock_openai_response(model: str = "gpt-4o") -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "chatcmpl_123", "model": model, "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "created": 0,
    }
    return resp


def _make_anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic"
    return client


def _make_openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai"
    return client


def test_record_anthropic_returns_correct_client():
    client = _make_anthropic_client()
    wrapped = record(client)
    assert isinstance(wrapped, AnthropicRecordedClient)


def test_record_openai_returns_correct_client():
    client = _make_openai_client()
    wrapped = record(client)
    assert isinstance(wrapped, OpenAIRecordedClient)


def test_record_unknown_raises():
    client = MagicMock()
    client.__class__.__module__ = "somelib"
    with pytest.raises(ValueError, match="Unsupported"):
        record(client)


def test_anthropic_messages_create_records(tmp_path):
    client = _make_anthropic_client()
    client.messages.create.return_value = _mock_anthropic_response()
    wrapped = AnthropicRecordedClient(client)
    with patch("recall.providers.anthropic.save"):
        wrapped.messages.create(model="claude-sonnet-4-6", messages=[])
    assert len(wrapped._session.events) == 2
    assert wrapped._session.events[0].type == "request"
    assert wrapped._session.events[1].type == "response"


def test_anthropic_model_extracted():
    client = _make_anthropic_client()
    client.messages.create.return_value = _mock_anthropic_response("claude-opus-4-7")
    wrapped = AnthropicRecordedClient(client)
    with patch("recall.providers.anthropic.save"):
        wrapped.messages.create(model="claude-opus-4-7", messages=[])
    assert wrapped._session.model == "claude-opus-4-7"


def test_anthropic_error_recorded_and_reraised():
    client = _make_anthropic_client()
    client.messages.create.side_effect = RuntimeError("API down")
    wrapped = AnthropicRecordedClient(client)
    with patch("recall.providers.anthropic.save"):
        with pytest.raises(RuntimeError):
            wrapped.messages.create(model="m", messages=[])
    assert wrapped._session.events[-1].type == "error"


def test_openai_chat_completions_create_records():
    client = _make_openai_client()
    client.chat.completions.create.return_value = _mock_openai_response()
    wrapped = OpenAIRecordedClient(client)
    with patch("recall.providers.openai.save"):
        wrapped.chat.completions.create(model="gpt-4o", messages=[])
    assert len(wrapped._session.events) == 2
    assert wrapped._session.events[0].type == "request"
    assert wrapped._session.events[1].type == "response"
    assert wrapped._session.provider == "openai"


def test_passthrough_attribute():
    client = _make_anthropic_client()
    client.some_attr = "value"
    wrapped = record(client)
    assert wrapped.some_attr == "value"
