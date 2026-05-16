"""Replay demo: replay a recorded session with no API calls, then patch a response."""
import sys
from pathlib import Path

import anthropic

from recall import record, replay
from recall.storage import list_sessions

# Step 1: find the most recent real session
sessions = list_sessions()
if not sessions:
    print("No sessions found. Run demo.py first.")
    sys.exit(1)

session_path = sessions[-1]
print(f"Replaying: {session_path.name}")
print()

# Step 2: replay - zero API calls, same responses
client = replay(anthropic.Anthropic(), session_path)
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print("Replayed response:", response.content[0].text)
print()

# Step 3: replay with a patched response - test what happens if the model had said something else
client2 = replay(
    anthropic.Anthropic(),
    session_path,
    patches={
        0: {
            "content": [{"type": "text", "text": "Greetings, human. I am a patched response.", "citations": None}]
        }
    },
)
response2 = client2.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print("Patched response:", response2.content[0].text)
print()
print("No API calls were made during replay.")
