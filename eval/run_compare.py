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
- Before any call, the number of calls still to make and the worst-case LLM
  calls and tokens (calls x the per-run budget) are printed; ``--estimate``
  stops there. A batch on a free tier is bounded by that budget, so it is
  known before it starts, not discovered on the bill or the rate limit.
- Each record carries the provider, model and cache setting it ran with, so a
  results file says what produced it even after the defaults change.
- ``--workers N`` runs N calls at a time, each worker holding ONE Groq key, so
  parallel calls never share a per-key rate limit (N is capped at the number of
  keys). The latency stored in each record is still that call's own wall time,
  measured inside ``answer_question`` exactly as with one worker; only the
  batch finishes sooner. Records are appended under a lock, in completion
  order, which the idempotency keys make harmless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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


def run_settings() -> dict:
    """Provider, model, cache flag and budget of this process, stored with every record."""
    from rr import settings

    model = settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.OLLAMA_MODEL
    return {"provider": settings.LLM_PROVIDER, "model": model, "llm_cache": settings.LLM_CACHE,
            "max_steps": settings.MAX_STEPS, "max_tokens": settings.MAX_TOKENS}


def estimate(pending: int, max_steps: int, max_tokens: int) -> str:
    """One line: calls still to make and their worst case in LLM calls and tokens."""
    return (f"{pending} call(s) to make; at most {pending * max_steps} LLM calls and "
            f"{pending * max_tokens:,} tokens (budget {max_steps} steps / {max_tokens:,} tokens per call)")


def _default_answer_fn() -> Callable[..., dict]:
    """Import the real ``answer_question`` only when a real run is requested."""
    from rr.run import answer_question

    return answer_question


def make_worker_llms(workers: int) -> list:
    """One ChatLLM per worker, each pinned to a single Groq key (capped at the key count).

    Two workers on the same key would both run into that key's rate limit and
    wait on it in turn, which is slower than one worker; with Ollama there is no
    key, so the requested number of workers is kept.
    """
    from rr.llm import ChatLLM

    base = ChatLLM()
    if base.provider != "groq":
        return [ChatLLM() for _ in range(max(1, workers))]
    n = max(1, min(workers, len(base.keys)))
    llms = []
    for i in range(n):
        llm = ChatLLM()
        llm.keys = [base.keys[i]]
        llms.append(llm)
    return llms


def run(
    run_id: str,
    questions: list[dict],
    systems: tuple[str, ...] = SYSTEMS,
    repeats: int = 3,
    runs_dir: Path = RUNS_DIR,
    answer_fn: Callable[..., dict] | None = None,
    sleep_s: float = 0.0,
    log: Callable[[str], None] = print,
    run_meta: dict | None = None,
    llms: list | None = None,
) -> dict:
    """Execute every missing (system, question, repeat) call and append its record.

    `run_meta` (provider, model, ...) is added to every record. `llms` is the
    list of clients to run with: one entry (or None) runs the calls one after
    the other; several run that many calls at a time, each call borrowing one
    client (see the module docstring). Returns counters {done, skipped, failed}
    so callers and tests can check resume behaviour without re-reading the files.
    """
    answer_fn = answer_fn or _default_answer_fn()
    runs_path = Path(runs_dir) / f"{run_id}.jsonl"
    dead_path = Path(runs_dir) / "dead_letter.jsonl"
    seen = done_keys(runs_path)
    stats = {"done": 0, "skipped": 0, "failed": 0}
    run_meta = run_meta or {}
    lock = threading.Lock()
    clients: queue.Queue = queue.Queue()
    for llm in (llms or [None]):
        clients.put(llm)

    def one(job: tuple) -> None:
        repeat, q, system = job
        key = record_key(system, q["id"], repeat, run_id)
        with lock:
            if key in seen:
                stats["skipped"] += 1
                return
        meta = {"key": key, "run_id": run_id, "system": system, "qid": q["id"],
                "repeat": repeat, "type": q["type"]}
        llm = clients.get()
        try:
            record = answer_fn(system, q["question"], run_id=run_id, qid=q["id"], llm=llm,
                               injected_passage=q.get("injected_passage"))
        except Exception as exc:  # any failure must be parked, not crash the batch
            with lock:
                stats["failed"] += 1
                _append(dead_path, {**meta, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(limit=5),
                                    "at": datetime.now(timezone.utc).isoformat()})
            log(f"[dead] {system} {q['id']} r{repeat}: {exc}")
        else:
            with lock:
                _append(runs_path, {**record, **run_meta, **meta})
                seen.add(key)
                stats["done"] += 1
            log(f"[ok]   {system} {q['id']} r{repeat}")
        finally:
            clients.put(llm)
        if sleep_s:
            time.sleep(sleep_s)

    jobs = [(repeat, q, system) for repeat in range(repeats) for q in questions for system in systems]
    workers = len(llms) if llms else 1
    if workers == 1:
        for job in jobs:
            one(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, jobs))
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
    p.add_argument("--estimate", action="store_true",
                   help="Print the calls still to make and their worst-case cost, then stop.")
    p.add_argument("--workers", type=int, default=1,
                   help="Calls to run at a time, one Groq key each (capped at the key count).")
    args = p.parse_args(argv)

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        p.error(f"unknown system(s): {', '.join(sorted(unknown))}")
    questions = load_questions(args.questions, args.qtype, args.limit)
    meta = run_settings()
    seen = done_keys(Path(args.runs_dir) / f"{args.run_id}.jsonl")
    pending = sum(record_key(s, q["id"], r, args.run_id) not in seen
                  for r in range(args.repeats) for q in questions for s in systems)
    print(f"run {args.run_id} with {meta['provider']} {meta['model']} (cache "
          f"{'on' if meta['llm_cache'] else 'off'}): "
          + estimate(pending, meta["max_steps"], meta["max_tokens"]))
    if args.estimate:
        return
    llms = make_worker_llms(args.workers) if args.workers > 1 else None
    meta["workers"] = len(llms) if llms else 1
    if llms and len(llms) < args.workers:
        print(f"only {len(llms)} worker(s): one Groq key each, and that many keys are configured")
    stats = run(args.run_id, questions, systems, args.repeats, args.runs_dir, sleep_s=args.sleep,
                run_meta=meta, llms=llms)
    print(f"run {args.run_id}: {stats}  -> {Path(args.runs_dir) / (args.run_id + '.jsonl')}")


if __name__ == "__main__":
    main()
