# Running an eval

Needs grounded-rag next to this folder (or `GROUNDED_RAG_PATH`), checked out at
the commit pinned in `.github/workflows/ci.yml`, with both indexes built, and the
Groq keys in `.env`. The published runs used `openai/gpt-oss-120b`, while
`.env.example` defaults to `openai/gpt-oss-20b`, and the LLM cache must be off
(`RR_LLM_CACHE=0`), otherwise the repeats replay the first answer. The model is
named with `--models`, which also gives the file its
`<run id>--groq-openai-gpt-oss-120b.jsonl` name.

**Launch.** Check the worst-case cost first, then run. One worker per Groq key.

```bash
export RR_LLM_CACHE=0
python -m eval.run_compare --run-id r1 --repeats 3 --models openai/gpt-oss-120b --estimate
python -m eval.run_compare --run-id r1 --repeats 3 --models openai/gpt-oss-120b --workers 3
```

**Search-depth ablation.** Same command with the agent only and the workflow's
depth, on the same pin as the run it is compared with:

```bash
RR_MAX_SEARCH_K=4 python -m eval.run_compare --run-id r2 --repeats 3 --systems agent --models openai/gpt-oss-120b
python -m eval.check_run runs/r2--groq-openai-gpt-oss-120b.jsonl --expected 120
```

**Resume.** Reuse the run id. Records already in `runs/r1--groq-openai-gpt-oss-120b.jsonl` are skipped,
so an interrupted run picks up where it stopped. A call that fails after its
retries goes to `runs/dead_letter.jsonl` and is retried on the next launch.

**Check.** The run is finished only when the file holds 240 records (40
questions x 2 systems x 3 repeats), no (system, question, repeat) twice, and
every record came from Groq:

```bash
python -m eval.check_run runs/r1--groq-openai-gpt-oss-120b.jsonl   # exit 0 only if complete; --expected N for another size
```

**Provenance.** Every record carries `code_sha` and `code_dirty` (route-or-roam,
ignoring `runs/` and `reports/`), `grounded_rag_sha` (`unknown` if
`GROUNDED_RAG_PATH` is not itself the root of a git checkout) with
`grounded_rag_dirty`, and `questions_sha256`, next to the model, cache flag,
budget and search depth. A run with `code_dirty: true` or
`grounded_rag_dirty: true` ran on code that is not in any commit.

**Report and cost.**

```bash
python -m eval.report --runs runs/r1--groq-openai-gpt-oss-120b.jsonl --out runs/r1.md    # report + chart next to it
python -m eval.cost runs/r1--groq-openai-gpt-oss-120b.jsonl --price-in 0.15 --price-out 0.60   # tokens and dollars per correct answer
```
