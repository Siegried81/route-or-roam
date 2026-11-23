"""Cost per successful answer, from runs files (tokens, and dollars when prices are given).

For each system: mean tokens in + out per record, divided by the pass rate
(``eval.score.score_record``, the same pass as the report). A run that fails
still spent its tokens, so the price of one good answer is the price of a
record over the share of records that pass. Infrastructure errors are left out
as in the report.

    python -m eval.cost runs/r125e--*.jsonl runs/r126e--*.jsonl [--price-in 0.15 --price-out 0.60]

Prices are dollars per million tokens and are never defaulted here: pass the
current ones from the provider's pricing page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.report import INFRA_ERROR_PREFIX, load_runs
from eval.run_compare import QUESTIONS_PATH, load_questions
from eval.score import score_record


def cost_per_success(runs: list[dict], questions: list[dict],
                     price_in: float | None = None, price_out: float | None = None) -> dict[str, dict]:
    """Per system: records, passes, mean tokens in/out, tokens per pass, and dollars per pass if priced."""
    by_id = {q["id"]: q for q in questions}
    acc: dict[str, dict] = {}
    for rec in runs:
        q = by_id.get(rec.get("qid"))
        if q is None or str(rec.get("error") or "").startswith(INFRA_ERROR_PREFIX):
            continue
        a = acc.setdefault(rec["system"], {"n": 0, "passed": 0, "tin": 0, "tout": 0})
        a["n"] += 1
        a["passed"] += score_record(q, rec)["passed"]
        a["tin"] += rec.get("tokens_in") or 0
        a["tout"] += rec.get("tokens_out") or 0
    out = {}
    for system, a in sorted(acc.items()):
        rate = a["passed"] / a["n"]
        row = {"n": a["n"], "passed": a["passed"], "pass_rate": rate,
               "tokens_in": a["tin"] / a["n"], "tokens_out": a["tout"] / a["n"]}
        row["tokens_per_pass"] = (row["tokens_in"] + row["tokens_out"]) / rate if rate else None
        if price_in is not None and price_out is not None and rate:
            row["usd_per_pass"] = (row["tokens_in"] * price_in + row["tokens_out"] * price_out) / 1e6 / rate
        out[system] = row
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: one block per runs file."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    p.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    args = p.parse_args(argv)
    questions = load_questions(args.questions)
    for path in args.runs:
        print(path.name)
        for system, r in cost_per_success(load_runs(path), questions, args.price_in, args.price_out).items():
            line = (f"  {system}: {r['passed']}/{r['n']} pass ({100 * r['pass_rate']:.1f}%), "
                    f"{r['tokens_in']:.0f} in + {r['tokens_out']:.0f} out per record, "
                    f"{r['tokens_per_pass']:,.0f} tokens per pass")
            if "usd_per_pass" in r:
                line += f", ${r['usd_per_pass']:.5f} per pass"
            print(line)


if __name__ == "__main__":
    main()
