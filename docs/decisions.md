# Decisions

One dated entry per change that alters what the system promises. Append, never rewrite.

## 2026-10-10 — Approving over the API needs an identity

- **What:** `POST /api/approve` checks who calls it. With `RR_APPROVE_TOKEN`
  unset, only requests from this machine may approve (403 otherwise); with it
  set, the `X-Approve-Token` header must match it, compared in constant time
  (401 otherwise). The response carries `decided_by`. The three other routes
  are unchanged: they read, approving writes.
- **Why:** the human gate is the agent's one irreversible action, and the API
  accepted the decision from anyone who could reach the port. On localhost
  that is the owner; started with `--host 0.0.0.0`, it was anyone on the
  network, writing a report in the owner's name.
- **What it does not settle:** a token is one shared secret, not a user. If
  the API is ever exposed to several people, replace it with a session and a
  name in `decided_by`.
- **Revisit if:** the React UI is deployed off-box — it then has to send the
  header (`web/src/AskTab.jsx`, the `api("/api/approve", …)` call).

## 2026-10-10 — Deployable as one Docker image, grounded-rag cloned inside

- **What:** `Dockerfile` (Node stage for the UI, Python runtime with grounded-rag
  cloned at a pinned ref into `/opt/grounded-rag`), `docker/entrypoint.sh`
  (optional index build at start, uvicorn on `$PORT`), `docker-compose.yml`
  (API, one-shot `ingest`, bundled Ollama, `rag_index` volume), `render.yaml`
  (hosted embedder, indexes rebuilt on each start). The UI sends
  `X-Approve-Token` from a field that appears only once the server asks for it.
- **Why:** route-or-roam reaches grounded-rag through `sys.path` and uses its
  corpora and indexes, so "install the package" is not a deployment; a checkout
  at a known ref is. Without the header the identity check added the same day
  would have locked the deployed UI out of approving.
- **What it does not settle:** the Render blueprint is written from the
  platform's documented behaviour, not from a deploy; the first deploy is the
  test. Rebuilding indexes on every start costs embedding calls each deploy,
  acceptable for two small corpora, not for a large one.
- **Revisit if:** grounded-rag gains a `pyproject.toml` and ships its corpora
  as package data — the clone then becomes a pip install at a version.

## 2026-10-11 — Eval set relabelled against the full AI Act

- **What:** grounded-rag replaced its four-file French AI Act primer with the
  full Regulation (EU) 2024/1689 in English (309 files: recitals, articles,
  annexes). The 15 questions that touch the AI Act were relabelled against it
  (`eval/questions.jsonl`):
  - label only, same expected answer: sh07, sh09, mh07, cc01, cc03, cc04, cc06;
  - Annex III date, mh08, cc02, in03: 2 December 2027 becomes **2 August 2026**.
    The later date came from the proposed Digital Omnibus, which is not in the
    corpus; the regulation applies from 2 August 2026 and defers only Article
    6(1) to 2 August 2027. mh08 also drops "explainability", named only in a
    recital;
  - sh10 (why the Annex III obligations were postponed) is now **unanswerable**:
    nothing in the regulation mentions a postponement;
  - mh09 and cc05 lose their "16 months" half and ask instead for the date the
    GPAI obligations apply (2 August 2025) and the date prohibited practices
    were banned (2 February 2025);
  - sh08 asks for the two most tightly regulated categories (prohibited,
    high-risk): "limited" and "minimal risk" are commentary, not the text;
  - un03 (maximum fine) is now **answerable**: Article 99 gives EUR 35 000 000
    or 7 % of turnover, so its forbidden strings are gone.
  Type counts are unchanged (sh10 and un03 swap single_hop and unanswerable).
- **Why:** every AI Act gold source pointed at a file that no longer exists,
  so source recall read 0 and `test_string_facts_appear_in_gold_sources`
  failed against grounded-rag main. Keeping the old labels would have scored
  correct answers from the real regulation as wrong.
- **What it changes in the numbers:** pass rates, source recall and refusal
  counts measured before this date are **not comparable** with runs after it
  (`docs/results.md` predates it). The questions are still asked in French
  against an English corpus for the AI Act part.
- **How to catch the next drift:** `python -m eval.locate_gold` lists every
  gold source missing from the grounded-rag checkout and where its facts now
  live; it exits 1 while any is missing. It reports, it never relabels.

## 2026-10-12 — An opt-in LLM judge, reported next to the string score

- **What:** `eval/judge.py` makes one structured LLM call per stored record
  (fixed rubric, reply validated against a pydantic schema, no second call on
  an invalid reply) and writes the verdicts to their own jsonl.
  `eval.report --judge <file>` adds secondary rows: "success with judged
  facts" and the records where judge and strings disagree on the facts.
- **New number, not a changed one:** "success with judged facts" is success as
  defined in `eval/score.py` with criterion 1 (key facts) decided by the judge
  instead of string matching; citations, refusal, forbidden strings and errors
  are unchanged. The reference success rate, every existing row and every
  published figure stay as they are: without `--judge` the report is
  byte-identical (checked on the five stored runs files).
- **Why:** string matching rejects answers that are right in other words
  (`missed_hop`); a judge lists them without making the reference score depend
  on a model.
- **What it does not settle:** the judge is itself a model and has not been run
  on the published records, so its agreement with a human reader is unmeasured.
- **Revisit if:** the judge is run on the full set; read the "judge yes,
  strings no" list by eye before quoting the judged rate.

## 2026-10-13 — The fence catches spaced and fullwidth tags

- **What:** `rr/fence.py` matches `<untrusted_source` look-alikes on an NFKC
  copy of the passage, with whitespace allowed around the slash, and escapes
  only the matched spans; the source name is attribute-escaped. `scan()` runs
  on the NFKC form too.
- **Why:** `< /untrusted_source>` and the fullwidth `＜/untrusted_source＞`
  passed through unescaped, so a retrieved passage could close its own fence,
  and a source name holding a quote could add attributes to the tag
  (`source="evil.md" trusted="yes"`). Same fix as grounded-rag's and
  flood-policy-kit's fences.
- **Measurement change:** `injection_flags` now also counts passages that use
  those spellings, so it can only go up on the same corpus; runs from before
  this date are not strictly comparable on that column.

## 2026-10-14 — Results re-run on the full 40-question set, cache off

- **What:** `python -m eval.run_compare --run-id r120e --models
  openai/gpt-oss-120b --repeats 1` with `RR_LLM_CACHE=0`, then `eval.report`.
  The README table and `docs/results.md` now report this run.
- **Why:** the published 17-question results were labelled against a corpus that
  no longer exists and predated the agent's last-call step and refusal rule.
  The cache was turned off so no stored answer was replayed.
- **What changes in the numbers:** workflow 15/17 -> 28/40, agent 15/17 ->
  29/40; McNemar p = 1.000 in both. The agent spends 2.9x the input tokens and
  hits its budget on 10% of questions; four agent answers were lost to provider
  errors (two unparsable outputs, one tool call with tools disabled, one 413
  over the 8,000 TPM free-tier limit) and are scored as failures.
