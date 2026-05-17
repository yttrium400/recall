"""Tests for recall export --format markdown."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from recall.cli import app, _render_markdown
from recall.session import Event, Session
from recall.storage import save


runner = CliRunner()


def _session_with_tool_call() -> Session:
    s = Session(provider="anthropic", model="claude-sonnet-4-6")
    s.add_event(Event(seq=1, type="request", timestamp="2026-05-17T10:00:00Z", payload={
        "model": "claude-sonnet-4-6",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
    }))
    s.add_event(Event(seq=2, type="response", timestamp="2026-05-17T10:00:01Z", duration_ms=1200, payload={
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use",
        "stop_sequence": None, "stop_details": None, "container": None,
        "content": [
            {"type": "text", "text": "Let me check that for you."},
            {"type": "tool_use", "id": "tool_abc", "name": "get_weather", "input": {"location": "Tokyo"}},
        ],
        "usage": {
            "input_tokens": 50, "output_tokens": 30,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "server_tool_use": None, "service_tier": "standard", "inference_geo": "not_available",
        },
    }))
    s.add_event(Event(seq=3, type="request", timestamp="2026-05-17T10:00:02Z", payload={
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check that for you."},
                {"type": "tool_use", "id": "tool_abc", "name": "get_weather", "input": {"location": "Tokyo"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tool_abc", "content": "22°C and sunny"},
            ]},
        ],
    }))
    s.add_event(Event(seq=4, type="response", timestamp="2026-05-17T10:00:03Z", duration_ms=800, payload={
        "id": "msg_2", "type": "message", "role": "assistant",
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "stop_sequence": None, "stop_details": None, "container": None,
        "content": [{"type": "text", "text": "The weather in Tokyo is 22°C and sunny."}],
        "usage": {
            "input_tokens": 100, "output_tokens": 20,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "server_tool_use": None, "service_tier": "standard", "inference_geo": "not_available",
        },
    }))
    return s


def _simple_session() -> Session:
    s = Session(provider="openai", model="gpt-4o")
    s.add_event(Event(seq=1, type="request", timestamp="2026-05-17T09:00:00Z", payload={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
    }))
    s.add_event(Event(seq=2, type="response", timestamp="2026-05-17T09:00:01Z", duration_ms=500, payload={
        "id": "chatcmpl_1", "object": "chat.completion", "created": 0, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi there!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }))
    return s


# ---------------------------------------------------------------------------
# _render_markdown unit tests
# ---------------------------------------------------------------------------

def test_markdown_contains_session_metadata():
    md = _render_markdown(_simple_session())
    assert "# Recall Session Report" in md
    assert "gpt-4o" in md
    assert "openai" in md


def test_markdown_contains_messages():
    md = _render_markdown(_simple_session())
    assert "Hello!" in md
    assert "Hi there!" in md


def test_markdown_contains_call_headers():
    md = _render_markdown(_simple_session())
    assert "## Call #1" in md


def test_markdown_contains_summary():
    md = _render_markdown(_simple_session())
    assert "## Summary" in md
    assert "Total calls" in md
    assert "| 1 |" in md


def test_markdown_tool_call_rendered_as_code_block():
    md = _render_markdown(_session_with_tool_call())
    assert "**Tool call:** `get_weather`" in md
    assert "```json" in md
    assert '"location": "Tokyo"' in md


def test_markdown_tool_result_rendered():
    md = _render_markdown(_session_with_tool_call())
    assert "**Tool result:**" in md
    assert "22°C and sunny" in md


def test_markdown_system_prompt_rendered():
    md = _render_markdown(_session_with_tool_call())
    assert "**System:**" in md
    assert "You are a helpful assistant." in md


def test_markdown_token_counts_included():
    md = _render_markdown(_simple_session())
    assert "10 in / 5 out" in md


def test_markdown_cost_estimate_included():
    md = _render_markdown(_simple_session())
    assert "~$" in md


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_export_command_stdout(tmp_path):
    path = save(_simple_session(), base=tmp_path)
    result = runner.invoke(app, ["export", str(path)])
    assert result.exit_code == 0
    assert "# Recall Session Report" in result.output
    assert "Hello!" in result.output


def test_export_command_to_file(tmp_path):
    path = save(_simple_session(), base=tmp_path)
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["export", str(path), "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "# Recall Session Report" in content
    assert "Hello!" in content


def test_export_unsupported_format(tmp_path):
    path = save(_simple_session(), base=tmp_path)
    result = runner.invoke(app, ["export", str(path), "--format", "csv"])
    assert result.exit_code != 0


def test_export_missing_file(tmp_path):
    result = runner.invoke(app, ["export", str(tmp_path / "nonexistent.json")])
    assert result.exit_code != 0


def test_export_with_tool_calls_stdout(tmp_path):
    path = save(_session_with_tool_call(), base=tmp_path)
    result = runner.invoke(app, ["export", str(path)])
    assert result.exit_code == 0
    assert "get_weather" in result.output
    assert "22°C and sunny" in result.output
