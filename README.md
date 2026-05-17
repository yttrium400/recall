# recall

**A flight recorder for AI agents.**

Your agent ran 12 steps, failed on step 8, and you have no idea why. The logs show an exception. The LLM response was wrong. But you can't reproduce it - the next run is different.

**recall** records every LLM call your agent makes to a plain JSON file. Then lets you replay the session without spending tokens, or inject a different response at any step to test a fix.

```bash
pip install recall-agent
```

Works with **Anthropic** and **OpenAI** SDKs. No accounts. No infrastructure. Files you own.

---

## Record

One line to wrap your client. Everything else stays the same.

```python
import anthropic
from recall import record

client = record(anthropic.Anthropic())

# your agent code, unchanged
messages = [{"role": "user", "content": "Analyze this bug report..."}]

r1 = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=messages)
messages.append({"role": "assistant", "content": r1.content})
messages.append({"role": "user", "content": "What caused it?"})

r2 = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=messages)
```

Session saved to `.recall/sessions/2026-05-16T10:32:00_abc123.json`.

OpenAI works the same way:

```python
from openai import OpenAI
from recall import record

client = record(OpenAI())
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Streaming is also supported for both providers.

---

## Inspect

```bash
# conversation view - see exactly what the model saw and said, call by call
recall play .recall/sessions/session.json

# step through one call at a time
recall play session.json --step

# raw JSON events if you need them
recall play session.json --raw

# token usage, cost estimate, latency breakdown
recall stats session.json

# list all sessions
recall ls
```

`recall play` shows the actual dialogue, not JSON blobs:

```
Call #1  10:32:00
  user: Analyze this bug report: NoneType on line 47...
  assistant: The error originates from get_user() returning None...
  tokens: 147 in / 300 out  ~$0.0013  1.2s

Call #2  10:32:02
  user: What caused it?
  assistant: payments-api v2.1.4 removed the null check introduced in v2.0.8...
  tokens: 469 in / 241 out  ~$0.0025  0.9s
```

---

## Replay

The part that doesn't exist anywhere else for raw SDK users.

```python
from recall import replay

# re-run your agent - zero API calls, same responses
client = replay(anthropic.Anthropic(), "session.json")

r1 = client.messages.create(...)  # returns the recorded response instantly
messages.append({"role": "assistant", "content": r1.content})

r2 = client.messages.create(...)  # same
```

Your agent code runs unchanged. Every `messages.create()` call returns the exact response from the recording. Deterministic, instant, free.

### Patch a response to test a fix

```python
# What if the model had said something different at step 1?
client = replay(
    anthropic.Anthropic(),
    "session.json",
    patches={
        0: {"content": [{"type": "text", "text": "The root cause is a missing null check."}]}
    },
)
```

Now your agent sees a different response at call #0. Subsequent calls still return their recorded responses. Useful for testing: "given this corrected analysis at step 1, does the rest of my agent handle it correctly?"

> **Note on multi-turn patching**: patching call N changes the context your agent builds for call N+1 onward. Calls N+1, N+2... still return their *recorded* responses, which were generated from the original context. For full re-simulation from the patch point, run the real agent with `record()` from that point.

### Compare two runs

```bash
recall diff session_broken.json session_working.json
```

Pinpoints the call where the sessions diverged and shows both responses side by side.

---

## Session format

Plain JSON. No proprietary format. No database. Version it in git. Share it with a colleague. Parse it with `jq`.

```json
{
  "id": "ca4a2165-...",
  "started_at": "2026-05-16T10:32:00Z",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "events": [
    {
      "seq": 1,
      "type": "request",
      "timestamp": "...",
      "payload": { "model": "...", "messages": [...] }
    },
    {
      "seq": 2,
      "type": "response",
      "timestamp": "...",
      "duration_ms": 1240,
      "payload": { "id": "msg_...", "content": [...], "usage": {...} }
    }
  ]
}
```

---

## Why not LangSmith / Langfuse / W&B Weave?

Those tools are great if you're already in their ecosystem. recall is for developers who are making raw `anthropic.Anthropic()` or `OpenAI()` calls, don't want to sign up for anything, and just need to understand why their agent is failing.

| | recall | LangSmith / Langfuse |
|---|---|---|
| Setup | `pip install recall-agent` | Account + API key + SDK wrapper |
| Storage | Local JSON files you own | Cloud platform |
| Framework | Any (raw SDK calls) | LangChain-first |
| Replay | Built-in | No |
| Cost | Free | Free tier, then pricing |

---

## Install

```bash
pip install recall-agent
```

Python 3.10+. Runtime dependencies: `typer`, `rich`. Bring your own `anthropic` or `openai` SDK.

---

## Quickstart demo

```bash
git clone https://github.com/yttrium400/recall.git
cd recall
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# set your API key
export ANTHROPIC_API_KEY=...

# run the multi-turn agent demo
python demo_agent.py

# inspect the recorded session
recall ls
recall play .recall/sessions/<session-file>.json
recall stats .recall/sessions/<session-file>.json
```

---

## Roadmap

- [x] Anthropic SDK - batch + streaming
- [x] OpenAI SDK - batch + streaming
- [x] `recall play` - conversation view, delta-only per call
- [x] `recall stats` - tokens, cost estimate, latency
- [x] `recall ls` - session list with cost summary
- [x] `recall diff` - compare sessions, highlight divergence
- [x] `recall replay` - deterministic replay, zero API calls
- [x] Response patching - inject different responses to test fixes
- [x] `recall export --format markdown` - shareable session reports
- [x] Async client support (`AsyncAnthropic`, `AsyncOpenAI`)
- [x] Tool call visualization in play view

---

## Contributing

```bash
git clone https://github.com/yttrium400/recall.git
cd recall && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
