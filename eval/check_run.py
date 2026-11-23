"""Check whether an eval runs file is complete.

Exit 0 only when the file has the expected number of records (240 by default:
40 questions x 2 systems x 3 repeats), no (system, qid, repeat) appears twice
and every record came from the expected provider. A wrapper that restarts an
interrupted run can loop until this passes, so a duplicate or a silent provider
fallback keeps a run from being called finished.

    python -m eval.check_run runs/r1.jsonl [--expected 240] [--provider groq]
    python -m eval.check_run runs/r1.jsonl --questions eval/questions_v2.jsonl --repeats 2

``--questions`` derives the expected count from a questions file (questions x
systems x ``--repeats``, 2 systems and 3 repeats by default), so a 100-question
run is not checked against 240. ``--expected`` wins over it; with neither flag
the default is still 240.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def check(path: Path, expected: int = 240, provider: str = "groq") -> tuple[bool, str]:
    """Return (complete, one-line summary) for the runs file at ``path``."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = Counter((r["system"], r["qid"], r["repeat"]) for r in rows)
    dupes = [k for k, n in keys.items() if n > 1]
    providers = Counter(r.get("provider") for r in rows)
    line = f"records={len(rows)}/{expected} duplicates={len(dupes)} providers={dict(providers)}"
    if dupes:
        line += f" duplicate keys: {dupes[:10]}"
    return len(rows) == expected and not dupes and set(providers) == {provider}, line


def expected_from_questions(path: Path, repeats: int = 3, systems: int = 2) -> int:
    """Records a complete run over the questions file at ``path`` holds."""
    n = sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return n * systems * repeats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; the exit code is 0 only for a complete, clean file."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_file", type=Path)
    p.add_argument("--expected", type=int, default=None,
                   help="Records expected (default: 240, or derived from --questions).")
    p.add_argument("--questions", type=Path, default=None,
                   help="Questions file the run used: expected = questions x 2 systems x --repeats.")
    p.add_argument("--repeats", type=int, default=3, help="Repeats per question, with --questions.")
    p.add_argument("--provider", default="groq")
    args = p.parse_args(argv)
    expected = args.expected
    if expected is None:
        expected = expected_from_questions(args.questions, args.repeats) if args.questions else 240
    ok, line = check(args.runs_file, expected, args.provider)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
