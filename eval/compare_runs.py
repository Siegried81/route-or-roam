"""Compare two or three runs question by question: is a gap retrieval, or chance?

Usage:
  python -m eval.compare_runs RUN_A RUN_B [RUN_C] [--questions eval/questions.jsonl]

With two runs it prints, per system, the passes of each run, how many records
keep the same verdict, and the majority-over-repeats counts (a question passes
when it passed in a strict majority of its repeats, as `eval.report` does).

With a third run C it also lists every (system, question) pair on which A and
B disagree at the majority, and says which side C lands on. The intended use is
a control: B ran new code or a new retrieval, C reruns B's code under A's
retrieval. If C sides with B, the gap between A and B is mostly chance (the luck
of A's draw); if C sides with A, the retrieval explains it. Counting such pairs
only separates the two causes roughly, and a handful of pairs cannot size an
effect: read the list, not just the tally.

Records are scored with `eval.score.score_record`, so the verdicts are the ones
the report would give. Only cells present in both runs are compared, and a
majority is only taken when all of a run's repeats are in. No LLM call.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from eval.score import score_record
from eval.stats import mcnemar_exact, wilson

SYSTEMS = ("workflow", "agent")


def load_questions(path: str | Path) -> dict[str, dict]:
    """Questions keyed by id."""
    with open(path, encoding="utf-8") as fh:
        return {q["id"]: q for q in map(json.loads, (ln for ln in fh if ln.strip()))}


def load_run(path: str | Path, questions: dict[str, dict]) -> dict:
    """Return {(system, qid, repeat): (passed, refused)}, scored like eval.report."""
    cells = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("qid") not in questions:
                continue
            scored = score_record(questions[rec["qid"]], rec)
            cells[(rec["system"], rec["qid"], int(rec["repeat"]))] = (
                bool(scored["passed"]), bool(rec.get("refused")))
    return cells


def repeats_of(cells: dict) -> int:
    """Number of repeats a run holds (highest repeat index + 1)."""
    return max((k[2] for k in cells), default=-1) + 1


def majority(cells: dict, questions: dict) -> dict:
    """{(system, qid): passed} for questions with all repeats in; more than half wins."""
    n = repeats_of(cells)
    out = {}
    for system in SYSTEMS:
        for qid in questions:
            draws = [cells[(system, qid, i)][0] for i in range(n) if (system, qid, i) in cells]
            if n and len(draws) == n:
                out[(system, qid)] = sum(draws) * 2 > n
    return out


def pattern(cells: dict, system: str, qid: str) -> str:
    """One letter per repeat: P passed, R refused, x failed otherwise, . missing."""
    n = repeats_of(cells)
    return "".join(
        "." if (system, qid, i) not in cells
        else "P" if cells[(system, qid, i)][0] else "R" if cells[(system, qid, i)][1] else "x"
        for i in range(n))


def landing(a: dict, b: dict, c: dict, questions: dict) -> list[tuple]:
    """Pairs where A and B disagree at the majority: (system, qid, side C lands on).

    The side is "A", "B", or "pending" when C has no full majority for the pair.
    """
    ma, mb, mc = majority(a, questions), majority(b, questions), majority(c, questions)
    rows = []
    for system in SYSTEMS:
        for qid in sorted(questions):
            k = (system, qid)
            if k in ma and k in mb and ma[k] != mb[k]:
                side = "pending" if k not in mc else "A" if mc[k] == ma[k] else "B"
                rows.append((system, qid, side))
    return rows


def report(paths: list[str], questions: dict) -> str:
    """The text the command prints."""
    runs = [load_run(p, questions) for p in paths]
    names = [Path(p).name for p in paths]
    out = []
    for label, name, cells in zip("ABC", names, runs):
        out.append(f"{label} = {name}: {len(cells)} records, {repeats_of(cells)} repeat(s)")
    a, b = runs[0], runs[1]
    out.append("\n== Same cells in A and B")
    for system in SYSTEMS:
        ks = [k for k in b if k[0] == system and k in a]
        same = sum(a[k][0] == b[k][0] for k in ks)
        out.append(f"  {system:8} n={len(ks)}  A {sum(a[k][0] for k in ks)}  "
                   f"B {sum(b[k][0] for k in ks)}  same verdict {same}/{len(ks)}")
    for label, cells in zip("ABC", runs):
        m = majority(cells, questions)
        out.append(f"\n== Majority over repeats, run {label} ({len(m)}/{2 * len(questions)} pairs)")
        for system in SYSTEMS:
            k = sum(v for (s, _), v in m.items() if s == system)
            n = sum(1 for (s, _) in m if s == system)
            lo, hi = wilson(k, n) if n else (0.0, 0.0)
            out.append(f"  {system:8} {k}/{n}  Wilson {lo:.2f}-{hi:.2f}")
        if len(m) == 2 * len(questions):
            wf = sum(m[("workflow", q)] and not m[("agent", q)] for q in questions)
            ag = sum(m[("agent", q)] and not m[("workflow", q)] for q in questions)
            out.append(f"  workflow only {wf}, agent only {ag}, McNemar p = {mcnemar_exact(wf, ag):.4f}")
    if len(runs) == 3:
        c = runs[2]
        rows = landing(a, b, c, questions)
        out.append("\n== Pairs where A and B disagree at the majority: where does C land?")
        for system, qid, side in rows:
            out.append(f"  {system:8} {qid:5} A {pattern(a, system, qid)}  B {pattern(b, system, qid)}"
                       f"  C {pattern(c, system, qid)}  -> {side}")
        tally = collections.Counter(side for _, _, side in rows)
        out.append("  " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare two or three runs question by question.")
    ap.add_argument("runs", nargs="+", help="two or three runs files (A, B, optional control C)")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    args = ap.parse_args(argv)
    if not 2 <= len(args.runs) <= 3:
        ap.error("give two or three runs files")
    print(report(args.runs, load_questions(args.questions)))


if __name__ == "__main__":
    main()
