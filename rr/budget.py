"""Per-run budget: LLM steps and tokens, shared by both systems.

A step is one LLM call, whichever system makes it, so the workflow (a fixed
number of calls) and the agent (a variable number) are held to the same limits.
The budget is plain data stored in graph state, so it survives a checkpoint and
resume across a human-approval pause.
"""

from __future__ import annotations

from rr import settings


def new_budget(max_steps: int | None = None, max_tokens: int | None = None) -> dict:
    """Return a fresh budget record with the configured (or given) limits."""
    return {
        "max_steps": max_steps or settings.MAX_STEPS,
        "max_tokens": max_tokens or settings.MAX_TOKENS,
        "steps": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }


def charge(budget: dict, tokens_in: int, tokens_out: int) -> dict:
    """Return a copy of `budget` with one more step and the call's tokens added."""
    return {**budget, "steps": budget["steps"] + 1,
            "tokens_in": budget["tokens_in"] + tokens_in,
            "tokens_out": budget["tokens_out"] + tokens_out}


def exhausted(budget: dict) -> str | None:
    """Name the limit that is used up ("max_steps" / "max_tokens"), or None if there is room.

    Checked before a call, so the last call may overshoot max_tokens by its own
    size; a hard pre-call cap would need a token estimate we do not trust.
    """
    if budget["steps"] >= budget["max_steps"]:
        return "max_steps"
    if budget["tokens_in"] + budget["tokens_out"] >= budget["max_tokens"]:
        return "max_tokens"
    return None


#: Share of the token budget after which the next call is treated as the last one.
LAST_CALL_TOKEN_SHARE = 0.8


def nearly_exhausted(budget: dict) -> bool:
    """True when the next call should be the run's last: one step left, or 80% of tokens used.

    Only meaningful while `exhausted` is still None. A loop-shaped system uses it
    to stop exploring and answer with what it has; it never adds a call.
    """
    used = budget["tokens_in"] + budget["tokens_out"]
    return (budget["steps"] >= budget["max_steps"] - 1
            or used >= LAST_CALL_TOKEN_SHARE * budget["max_tokens"])
