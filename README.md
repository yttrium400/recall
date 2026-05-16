# recall

**A flight recorder for AI agents.**

Record every LLM call your agent makes. Replay it later - deterministically, for free, with no API calls. Inject different responses to debug exactly where your agent went wrong.

```
pip install recall-agent
```

---

## The problem

Your agent ran 12 steps, then hallucinated a tool name and crashed. You have no idea:
- What the model actually saw at step 8
- Whether the failure was in your prompt, your tool results, or the model output
- How to reproduce it - the next run will be different

Existing tools (LangSmith, Langfuse, W&B Weave) solve this, but they require accounts, infrastructure, and are deeply coupled to specific frameworks.

**recall** is two functions and a CLI. It works with any LLM SDK. Sessions are plain JSON files you own locally.

---

## Record

One line to wrap your client. Everything else is unchanged.

```python
import anthropic
from recall import record

client = record(anthropic.Anthropic())

# use it exactly as before - every call is recorded
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
)
```

Session saved to `.recall/sessions/2026-05-16T10:32:00_abc123.json`.

---

## Inspect

```bash
# conversation view - see what the model actually saw and said
recall play .recall/sessions/session.json

# step through one call at a time
recall play session.json --step

# token usage, cost estimate, latency breakdown
recall stats session.json

# list all sessions with cost and token counts
recall ls
```

`recall play` renders the actual conversation - not raw JSON blobs:

```
Call #1  2026-05-16T10:32:00Z
  user: Summarize this document and extract action items.
  assistant: [tool_use: read_file] {"path": "report.pdf"}

Call #2  2026-05-16T10:32:04Z
  user: [tool_result] "Q3 revenue up 12%..."
  assistant: Here are the key action items: ...
  tokens: 1,240 in / 380 out  ~$0.009420  1,840ms
```

---

## Replay

This is the part that doesn't exist anywhere else.

```python
from recall import replay

# re-run your agent against the recorded session - zero API calls
client = replay(anthropic.Anthropic(), "session.json")
response = client.messages.create(...)  # returns the recorded response instantly
```

Your agent code runs unchanged. But instead of hitting the API, every call returns the exact response you recorded. Deterministic. Instant. Free.

### Patch a response to test a fix

```python
# what if the model had returned something different at call #2?
client = replay(
    anthropic.Anthropic(),
    "session.json",
    patches={
        1: {"content": [{"type": "text", "text": "I don't know how to do that."}]}
    },
)
```

Now your agent sees a different response at step 2. Does it handle it correctly? You find out in milliseconds, not minutes, and without spending tokens.

### Compare two runs

```bash
recall diff session_broken.json session_working.json
```

Pinpoints the exact call where the sessions diverged.

---

## Session format

Plain JSON. No proprietary format. No database.

```json
{
  "id": "7909d66c-94ca-4556-aae3-5c299f2460f0",
  "started_at": "2026-05-16T10:32:00Z",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "events": [
    {
      "seq": 1,
      "type": "request",
      "timestamp": "2026-05-16T10:32:00.123Z",
      "payload": { "model": "claude-sonnet-4-6", "messages": [...] }
    },
    {
      "seq": 2,
      "type": "response",
      "timestamp": "2026-05-16T10:32:01.963Z",
      "duration_ms": 1840,
      "payload": { "id": "msg_...", "content": [...], "usage": {...} }
    }
  ]
}
```

You can read it, version it in git, share it with a colleague, or parse it with `jq`.

---

## Install

```bash
pip install recall-agent
```

Requires Python 3.10+. Dependencies: `typer`, `rich`. That's it.

---

## Roadmap

- [x] Anthropic SDK wrapper
- [x] Session recording to plain JSON
- [x] `recall play` - conversation view
- [x] `recall stats` - tokens, cost, latency
- [x] `recall diff` - compare sessions, find divergence point
- [x] `recall replay` - deterministic replay, zero API calls
- [x] Response patching - inject different responses at any step
- [ ] OpenAI SDK support
- [ ] Streaming support
- [ ] `recall export --format markdown` - shareable session reports

---

## Contributing

```bash
git clone https://github.com/yttrium400/recall.git
cd recall
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
