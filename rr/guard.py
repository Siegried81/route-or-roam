"""Guard between the agent's tool requests and their execution.

The model proposes; the guard disposes. Each requested call gets a verdict:
"run", "approve" (a human must say yes first, e.g. save_report) or "block"
(an exact repeat of an earlier call, which can only return the same result).
A first repeat is blocked and the model is told why; a second one ends the run,
because a model that keeps repeating itself is looping, not progressing. Step
and token limits are enforced before each LLM call (see rr/budget.py), so the
guard only has to judge the calls themselves.
"""

from __future__ import annotations

import json

from rr.tools import TOOLS

MAX_LOOP_HITS = 2


def call_key(name: str, args: dict) -> str:
    """Canonical identity of a tool call: same name and same arguments, key order ignored."""
    return json.dumps([name, args], sort_keys=True, ensure_ascii=False, default=str)


def review(pending: list[dict], call_keys: list[str], loop_hits: int) -> dict:
    """Return verdicts for `pending` calls plus the updated loop bookkeeping.

    Result keys: pending (each call with a "verdict"), call_keys, loop_hits and
    stop_reason ("loop" once MAX_LOOP_HITS repeats were seen, else None).
    """
    keys = list(call_keys)
    reviewed = []
    for call in pending:
        key = call_key(call["name"], call["args"])
        if key in keys:
            verdict = "block"
            loop_hits += 1
        else:
            spec = TOOLS.get(call["name"])
            verdict = "approve" if spec and spec.needs_approval else "run"
            keys.append(key)
        reviewed.append({**call, "verdict": verdict})
    return {"pending": reviewed, "call_keys": keys, "loop_hits": loop_hits,
            "stop_reason": "loop" if loop_hits >= MAX_LOOP_HITS else None}
