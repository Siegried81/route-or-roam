# Running an eval

Needs grounded-rag next to this folder (or `GROUNDED_RAG_PATH`) with both
indexes built, and the Groq keys in `.env`.

**Launch.** Check the worst-case cost first, then run. One worker per Groq key.

```bash
python -m eval.run_compare --run-id r1 --repeats 3 --estimate
python -m eval.run_compare --run-id r1 --repeats 3 --workers 3
```

**Resume.** Reuse the run id. Records already in `runs/r1.jsonl` are skipped,
so an interrupted run picks up where it stopped. A call that fails after its
retries goes to `runs/dead_letter.jsonl` and is retried on the next launch.

**Check.** The run is finished only when the file holds 240 records (40
questions x 2 systems x 3 repeats), no (system, question, repeat) twice, and
every record came from Groq:

```bash
python -m eval.check_run runs/r1.jsonl        # exit 0 only if complete; --expected N for another size
```

**Provenance.** Every record carries `code_sha` and `code_dirty` (route-or-roam,
ignoring `runs/` and `reports/`), `grounded_rag_sha` (`unknown` if
`GROUNDED_RAG_PATH` is not a git checkout) and `questions_sha256`, next to the
model, cache flag, budget and search depth. A run with `code_dirty: true` ran on
code that is not in any commit.

**Report.**

```bash
python -m eval.report --runs runs/r1.jsonl    # docs/results.md + chart
```
