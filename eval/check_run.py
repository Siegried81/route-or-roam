"""Check whether an eval runs file is complete.

Exit 0 only when the file has the expected number of records (240 by default:
40 questions x 2 systems x 3 repeats), no (system, qid, repeat) appears twice
and every record came from the expected provider. A wrapper that restarts an
interrupted run can loop until this passes, so a duplicate or a silent provider
fallback keeps a run from being called finished.

    python -m eval.check_run runs/r1.jsonl [--expected 240] [--provider groq]
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; the exit code is 0 only for a complete, clean file."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_file", type=Path)
    p.add_argument("--expected", type=int, default=240)
    p.add_argument("--provider", default="groq")
    args = p.parse_args(argv)
    ok, line = check(args.runs_file, args.expected, args.provider)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
