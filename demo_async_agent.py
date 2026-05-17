"""
Async multi-turn agent demo — verifies async record/replay works end-to-end.

Uses claude-haiku-4-5-20251001 (cheapest Anthropic model).
Runs a 3-turn debugging conversation, replays it for free, then patches turn 2.
"""
import asyncio

import anthropic

from recall import record, replay

BUG_REPORT = """
Bug #4821 - Production Incident
Service: payments-api
Symptom: POST /charge returns 500 for ~2% of requests, started at 14:32 UTC
Error: "NoneType object has no attribute 'stripe_customer_id'"
Stack trace: user_service.get_user() -> charge_controller.py line 47
Recent deploys: payments-api v2.1.4 (14:15 UTC), user-service v3.0.1 (13:58 UTC)
"""

MODEL = "claude-haiku-4-5-20251001"


async def run_agent(client: any, verbose: bool = True) -> tuple[str, str, str]:
    """Three-turn async agent: triage -> root cause -> fix."""
    messages = [
        {"role": "user", "content": f"You are an expert SRE. Triage this bug:\n{BUG_REPORT}"}
    ]

    r1 = await client.messages.create(model=MODEL, max_tokens=200, messages=messages)
    triage = r1.content[0].text
    if verbose:
        print(f"\n[Turn 1 - Triage]\n{triage}\n")

    messages.append({"role": "assistant", "content": r1.content})
    messages.append({"role": "user", "content": "Which deploy is the likely culprit and why?"})

    r2 = await client.messages.create(model=MODEL, max_tokens=200, messages=messages)
    root_cause = r2.content[0].text
    if verbose:
        print(f"[Turn 2 - Root Cause]\n{root_cause}\n")

    messages.append({"role": "assistant", "content": r2.content})
    messages.append({"role": "user", "content": "Give me immediate remediation steps. Be brief."})

    r3 = await client.messages.create(model=MODEL, max_tokens=200, messages=messages)
    fix = r3.content[0].text
    if verbose:
        print(f"[Turn 3 - Fix]\n{fix}\n")

    return triage, root_cause, fix


async def main() -> None:
    # Step 1: Record with AsyncAnthropic
    print("=" * 60)
    print("STEP 1: Recording a 3-turn async agent session")
    print("=" * 60)

    recorded_client = record(anthropic.AsyncAnthropic())
    triage, root_cause, fix = await run_agent(recorded_client)
    session_file = recorded_client.session_path
    print(f"Session saved: {session_file.name}")

    # Step 2: Async replay — zero API calls
    print("\n" + "=" * 60)
    print("STEP 2: Replaying async (no API calls)")
    print("=" * 60)

    replay_client = replay(anthropic.AsyncAnthropic(), session_file)
    triage2, root_cause2, fix2 = await run_agent(replay_client, verbose=False)

    assert triage == triage2, "Turn 1 replay mismatch"
    assert root_cause == root_cause2, "Turn 2 replay mismatch"
    assert fix == fix2, "Turn 3 replay mismatch"
    print("All 3 turns replayed identically via async. Zero API calls made.")

    # Step 3: Patch turn 2 and verify downstream turn 3 still returns recorded response
    print("\n" + "=" * 60)
    print("STEP 3: Patching turn 2 response")
    print("=" * 60)

    alt_root_cause = (
        "The culprit is payments-api v2.1.4. The timing correlation is stronger and it directly "
        "calls user_service.get_user(). The new version likely stopped handling None returns."
    )
    patched_client = replay(
        anthropic.AsyncAnthropic(),
        session_file,
        patches={1: {"content": [{"type": "text", "text": alt_root_cause, "citations": None}]}},
    )
    _, patched_rc, patched_fix = await run_agent(patched_client, verbose=False)
    assert patched_rc == alt_root_cause, "Patch was not applied to turn 2"
    assert patched_fix == fix, "Turn 3 should still return recorded response"
    print(f"Patched root cause applied correctly.")
    print(f"Turn 3 still returned the original recorded response (as expected).")

    print("\n" + "=" * 60)
    print(f"recall play .recall/sessions/{session_file.name}")
    print(f"recall stats .recall/sessions/{session_file.name}")
    print(f"recall export .recall/sessions/{session_file.name}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
