"""
Multi-turn agent demo showing the full recall debugging workflow.

Scenario: A code review agent that:
  1. Reads a bug report (multi-turn, uses context from prior turns)
  2. Identifies the root cause
  3. Suggests a fix

We record the session, inspect it, replay it for free, then patch
one response to simulate "what if the model had identified a different cause?"
"""
import json
import sys

import anthropic

from recall import record, replay
from recall.storage import list_sessions

BUG_REPORT = """
Bug #4821 - Production Incident
Service: payments-api
Symptom: POST /charge returns 500 for ~2% of requests, started at 14:32 UTC
Error in logs: "NoneType object has no attribute 'stripe_customer_id'"
Stack trace points to: user_service.get_user() -> charge_controller.py line 47
Recent deploys: payments-api v2.1.4 (14:15 UTC), user-service v3.0.1 (13:58 UTC)
"""


def run_agent(client: any, verbose: bool = True) -> tuple[str, str, str]:
    """Three-turn agent: triage -> root cause -> fix recommendation."""
    messages = [
        {"role": "user", "content": f"You are an expert SRE. Triage this bug:\n{BUG_REPORT}"}
    ]

    # Turn 1: initial triage
    r1 = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=messages,
    )
    triage = r1.content[0].text
    if verbose:
        print(f"\n[Turn 1 - Triage]\n{triage}\n")

    # Turn 2: deep root cause analysis
    messages.append({"role": "assistant", "content": r1.content})
    messages.append({"role": "user", "content": "Which of the two recent deploys is the likely culprit and why?"})

    r2 = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=messages,
    )
    root_cause = r2.content[0].text
    if verbose:
        print(f"[Turn 2 - Root Cause]\n{root_cause}\n")

    # Turn 3: fix recommendation
    messages.append({"role": "assistant", "content": r2.content})
    messages.append({"role": "user", "content": "Give me the immediate remediation steps. Be specific and brief."})

    r3 = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=messages,
    )
    fix = r3.content[0].text
    if verbose:
        print(f"[Turn 3 - Fix]\n{fix}\n")

    return triage, root_cause, fix


# ---------------------------------------------------------------------------
# Step 1: Record the session
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Recording a 3-turn agent session")
print("=" * 60)

recorded_client = record(anthropic.Anthropic())
triage, root_cause, fix = run_agent(recorded_client)

session_file = recorded_client.session_path
print(f"Session saved: {session_file.name}")

# ---------------------------------------------------------------------------
# Step 2: Replay - zero API calls, identical result
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Replaying the session (no API calls)")
print("=" * 60)

replay_client = replay(anthropic.Anthropic(), session_file)
triage2, root_cause2, fix2 = run_agent(replay_client, verbose=False)

assert triage == triage2 and root_cause == root_cause2 and fix == fix2
print("All 3 turns replayed identically. Zero API calls made.")

# ---------------------------------------------------------------------------
# Step 3: Patch turn 2 - simulate "what if model blamed the other deploy?"
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Patching turn 2 - testing a different root cause")
print("=" * 60)

alt_root_cause = (
    "The culprit is payments-api v2.1.4 (deployed at 14:15 UTC, 17 minutes before incident). "
    "The timing correlation is stronger, and payments-api directly calls user_service.get_user(). "
    "The new version likely stopped handling None returns from get_user() that user-service v3.0.1 "
    "now returns for recently-migrated accounts."
)

patched_client = replay(
    anthropic.Anthropic(),
    session_file,
    patches={
        1: {
            "content": [{"type": "text", "text": alt_root_cause, "citations": None}]
        }
    },
)

_, patched_root_cause, patched_fix = run_agent(patched_client, verbose=False)

print(f"Original root cause:\n  {root_cause[:120]}...\n")
print(f"Patched root cause:\n  {patched_root_cause[:120]}...\n")
print(f"Fix given original cause:\n  {fix[:120]}...\n")
print(f"Fix given patched cause:\n  {patched_fix[:120]}...\n")
print("Note: turn 3 used the *recorded* response (not re-called). To re-generate")
print("turn 3 based on the new root cause, run the real agent from that point.")

print("\n" + "=" * 60)
print("Run: recall play .recall/sessions/" + session_file.name)
print("Run: recall stats .recall/sessions/" + session_file.name)
print("=" * 60)
