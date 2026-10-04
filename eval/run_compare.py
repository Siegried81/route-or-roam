"""Run the workflow and the agent over the labelled questions, N repeats each.

Usage (from the project root):
    python -m eval.run_compare --run-id r1 [--repeats 3] [--systems workflow,agent]
                               [--type numeric] [--limit 5] [--sleep 2]

Why it is built this way:
- Every (system, qid, repeat, run_id) gets a sha256 key. Keys already present in
  runs/<run_id>.jsonl are skipped, so re-running the same command after a rate
  limit or a crash resumes exactly where it stopped and never double-counts.
- An exception from ``answer_question`` is not a result: it goes to
  runs/dead_letter.jsonl and is NOT written to the runs file, so the next
  resume retries it. A record that comes back with its own ``error`` field is a
  result (the system failed gracefully) and is kept and scored as such.
- Raw records are stored unscored; ``eval.report`` scores them against
  questions.jsonl, so a scoring fix never requires paying for a re-run.
- ``rr.run`` is imported lazily so this module (and its tests) load even when
  the systems package is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"
SYSTEMS = ("workflow", "agent")


def load_questions(path: Path = QUESTIONS_PATH, qtype: str | None = None, limit: int | None = None) -> list[dict]:
    """Read questions.jsonl, optionally keeping one type and the first ``limit`` items."""
    with Path(path).open(encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]
    if qtype:
        questions = [q for q in questions if q["type"] == qtype]
    return questions[:limit] if limit else questions


def record_key(system: str, qid: str, repeat: int, run_id: str) -> str:
    """Stable idempotency key for one call; identical inputs always give the same key."""
    return hashlib.sha256(f"{system}|{qid}|{repeat}|{run_id}".encode("utf-8")).hexdigest()


def done_keys(runs_path: Path) -> set[str]:
    """Keys already written to a runs file (empty set if the file does not exist yet).

    A truncated last line (process killed mid-write) is ignored rather than
    aborting the resume; that call simply runs again.
    """
    keys = set()
    if not runs_path.exists():
        return keys
    with runs_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _append(path: Path, obj: dict) -> None:
    """Append one JSON line and flush, so a crash loses at most the call in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def _default_answer_fn() -> Callable[..., dict]:
    """Import the real ``answer_question`` only when a real run is requested."""
    from rr.run import answer_question

    return answer_question


def run(
    run_id: str,
    questions: list[dict],
    systems: tuple[str, ...] = SYSTEMS,
    repeats: int = 3,
    runs_dir: Path = RUNS_DIR,
    answer_fn: Callable[..., dict] | None = None,
    sleep_s: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Execute every missing (system, question, repeat) call and append its record.

    Returns counters {done, skipped, failed} so callers and tests can check
    resume behaviour without re-reading the files.
    """
    answer_fn = answer_fn or _default_answer_fn()
    runs_path = Path(runs_dir) / f"{run_id}.jsonl"
    dead_path = Path(runs_dir) / "dead_letter.jsonl"
    seen = done_keys(runs_path)
    stats = {"done": 0, "skipped": 0, "failed": 0}

    for repeat in range(repeats):
        for q in questions:
            for system in systems:
                key = record_key(system, q["id"], repeat, run_id)
                if key in seen:
                    stats["skipped"] += 1
                    continue
                meta = {"key": key, "run_id": run_id, "system": system, "qid": q["id"],
                        "repeat": repeat, "type": q["type"]}
                try:
                    record = answer_fn(system, q["question"], run_id=run_id, qid=q["id"],
                                       injected_passage=q.get("injected_passage"))
                except Exception as exc:  # any failure must be parked, not crash the batch
                    stats["failed"] += 1
                    _append(dead_path, {**meta, "error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc(limit=5),
                                        "at": datetime.now(timezone.utc).isoformat()})
                    log(f"[dead] {system} {q['id']} r{repeat}: {exc}")
                else:
                    _append(runs_path, {**record, **meta})
                    seen.add(key)
                    stats["done"] += 1
                    log(f"[ok]   {system} {q['id']} r{repeat}")
                if sleep_s:
                    time.sleep(sleep_s)
    return stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; see the module docstring for the flags."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True, help="Name of the batch; reuse it to resume.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated: workflow,agent")
    p.add_argument("--type", dest="qtype", default=None, help="Only this question type.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions (after --type).")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between calls.")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = p.parse_args(argv)

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        p.error(f"unknown system(s): {', '.join(sorted(unknown))}")
    questions = load_questions(args.questions, args.qtype, args.limit)
    stats = run(args.run_id, questions, systems, args.repeats, args.runs_dir, sleep_s=args.sleep)
    print(f"run {args.run_id}: {stats}  -> {Path(args.runs_dir) / (args.run_id + '.jsonl')}")


if __name__ == "__main__":
    main()
