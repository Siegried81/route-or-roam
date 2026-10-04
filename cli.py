"""Command-line entry point: ask one question to the workflow or the agent.

    python cli.py ask --system workflow "What were Apple's net sales in FY2025?"
    python cli.py ask --system agent --approve "Summarise the AI Act timeline and save a report"

A thin wrapper over rr.run.answer_question, so the CLI exercises exactly what the
eval harness measures. save_report is rejected unless --approve is given, in
which case each request is shown and confirmed interactively.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from rr.run import answer_question


def _ask_human(payload: dict) -> bool:
    """Show a pending save_report call and ask for y/N on the terminal."""
    print(f"\nThe agent wants to call {payload['tool']}:")
    print(json.dumps(payload["args"], indent=2, ensure_ascii=False)[:2000])
    return input("Approve? [y/N] ").strip().lower() == "y"


def cmd_ask(args) -> int:
    """Run one question and print the answer, tool trace and run metrics."""
    res = answer_question(args.system, " ".join(args.question), run_id=f"cli-{uuid.uuid4().hex[:6]}",
                          qid="cli", approve=_ask_human if args.approve else None,
                          injected_passage=args.inject)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["error"] is None else 1
    print(res["answer"] or "(no answer)")
    print(f"\nCitations: {', '.join(res['citations']) or 'none'}")
    print(f"Sources searched: {', '.join(res['sources_searched']) or 'none'}")
    print("\nTool calls:")
    for c in res["tool_calls"]:
        status = "ok" if c["ok"] else f"FAILED ({c['error']})"
        print(f"  - {c['name']}({json.dumps(c['args'], ensure_ascii=False)[:120]}) {status}")
    print(f"\nLLM calls: {res['llm_calls']}  tokens: {res['tokens_in']} in / {res['tokens_out']} out  "
          f"latency: {res['latency_s']:.1f}s")
    flags = [k for k in ("refused", "budget_exhausted") if res[k]]
    flags += [f"{k}={res[k]}" for k in ("injection_flags", "hallucinated_tools") if res[k]]
    if flags:
        print("Flags: " + ", ".join(flags))
    if res["error"]:
        print(f"Error: {res['error']}")
        return 1
    return 0


def main() -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description="Route or Roam: workflow vs agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ask", help="answer one question with one system")
    p.add_argument("--system", choices=["workflow", "agent"], required=True)
    p.add_argument("--approve", action="store_true", help="ask before each save_report (default: reject)")
    p.add_argument("--inject", default=None, help="evaluation hook: passage appended to every search")
    p.add_argument("--json", action="store_true", help="print the raw result dict")
    p.add_argument("question", nargs="+")
    p.set_defaults(func=cmd_ask)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
