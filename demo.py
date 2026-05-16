"""End-to-end demo: wrap a real Anthropic client, make one API call, record the session."""
import anthropic

from recall import record

client = record(anthropic.Anthropic())

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print("Response:", response.content[0].text)
print()
print("Session saved. Run: recall ls")
