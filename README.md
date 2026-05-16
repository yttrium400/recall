# recall

A lightweight flight recorder for AI agent sessions. Records every LLM call to a plain JSON file you own locally.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import anthropic
from recall import record

client = record(anthropic.Anthropic())
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Session is saved to `.recall/sessions/` automatically.

## CLI

```bash
recall ls                          # list recorded sessions
recall play .recall/sessions/x.json        # render a session
recall play .recall/sessions/x.json --step # step through events
recall diff session_a.json session_b.json  # compare two sessions
```

## Run the demo

```bash
python demo.py
recall ls
recall play .recall/sessions/<filename>.json
```

## Tests

```bash
pytest
```
