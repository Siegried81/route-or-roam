"""Find where each question's key facts live in the corpus it is scored against.

``gold_sources`` are file names inside grounded-rag's corpora, and grounded-rag
is reused in place, so its corpus can change under this eval without a commit
here: replacing the four-file AI Act primer by the 309-file regulation left
every AI Act gold source pointing at a file that no longer exists, and
source recall silently reading 0 for them. This script is the check to run
whenever grounded-rag changes, and the mechanical half of the relabelling.

A fact counts as present in a file under ``score.fact_matched``, the same test
an answer is scored with, so "the corpus holds this fact" and "an answer
containing this fact passes" cannot drift apart. Numeric facts are skipped:
a small number such as 16 (months) matches nearly any file, and computed
answers (309 days) are in no file by design.

It reports and never rewrites. Picking the file a fact first appears in is not
labelling: "2024/1689" is quoted in Article 102 as well as in the title, and
"2 August 2026" in Article 57 as well as in Article 113, which is the one that
sets the date. A fact in no file is a change of answer, not of label - the
regulation says 2 August 2026 where the primer said 2 December 2027 - and that
is a decision for whoever owns the eval set. Exit code 1 while any gold
source is missing, so CI can run it.

Usage:
    python -m eval.locate_gold
    python -m eval.locate_gold --corpus ../grounded-rag/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.score import fact_matched
from rr.settings import GROUNDED_RAG_PATH, ROOT

QUESTIONS = ROOT / "eval" / "questions.jsonl"
CORPORA = ("filings_sections", "ai_act_sections")


def load_corpus(data_dir: Path) -> dict[str, str]:
    """File name -> text, over every corpus a gold source can point into."""
    corpus: dict[str, str] = {}
    for name in CORPORA:
        for path in sorted((data_dir / name).glob("*.txt")):
            corpus[path.name] = path.read_text(encoding="utf-8")
    return corpus


def locate(question: dict, corpus: dict[str, str]) -> dict:
    """Which gold sources are gone, and which files hold the facts they covered."""
    gold = question.get("gold_sources") or []
    kept = [s for s in gold if s in corpus]
    missing = [s for s in gold if s not in corpus]
    facts = [f for f in question.get("key_facts") or [] if isinstance(f, str)]
    uncovered = [f for f in facts if not any(fact_matched(f, corpus[s]) for s in kept)]
    candidates = {f: sorted(n for n, t in corpus.items() if fact_matched(f, t)) for f in uncovered}
    return {"id": question["id"], "missing": missing, "candidates": candidates,
            "numeric": [f for f in question.get("key_facts") or [] if not isinstance(f, str)]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=GROUNDED_RAG_PATH / "data")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    stale = [r for r in (locate(q, corpus) for q in questions) if r["missing"]]
    print(f"{len(corpus)} corpus files, {len(questions)} questions, {len(stale)} with a missing gold source")
    for r in stale:
        print(f"\n{r['id']}: gone {r['missing']}")
        for fact, files in r["candidates"].items():
            shown = ", ".join(files[:4]) + (f" (+{len(files) - 4})" if len(files) > 4 else "")
            print(f"  {fact[:48]:48} -> {shown or 'IN NO FILE'}")
        if r["numeric"]:
            print(f"  numeric facts, not located: {r['numeric']}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
