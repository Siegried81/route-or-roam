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

## 2026-11-06: Audit fixes, and three labels accept their French forms

- **Labels (`eval/questions.jsonl`, mirrored in `eval/paraphrases.jsonl`):**
  sh08 accepts "interdits/interdite/prohibé(e)(s)", mh07 accepts
  "subliminale(s)/subliminaux", mh09 accepts "fournisseur" (singular). These
  questions are asked in French, and a right answer written in French failed
  only because its inflection was missing from the alias list.
  - **What changes in the numbers:** the 14 October records, re-scored with no
    new LLM call, go from workflow 28/40 to 30/40 (mh07 and mh09: both answers
    are correct, read by eye). The agent stays at 29/40. No record changed
    from pass to fail. Paired: 5 workflow-only, 4 agent-only, McNemar p =
    1.000. README and `docs/results.md` report the re-scored figures.
- **Refusal (`rr/common.py::is_refusal`):** an answer is a refusal only when it
  is the refusal sentence and nothing else. Before, any answer that *contained*
  the sentence and cited nothing counted, so "Revenue was 416 billion. I
  couldn't find anything..." passed an unanswerable question. Curly
  apostrophes are now folded first. Measured on the 254 stored records: no
  `refused` value would change (every recorded refusal is the bare sentence).
- **Last call (`rr/budget.py::nearly_exhausted`):** also true when one more
  call as large as the previous one would reach the token cap. The agent's
  prompt grows by 3 to 4k tokens per search, so it could jump past 100% from
  under 80% and stop on `max_tokens` with the budget message. That is what
  happened to agent cc06, in03 and mh08 on 14 October (its three
  `loop_or_budget` failures). It changes what `budget_exhausted` means for the
  agent: future runs should show fewer budget stops and more last-call answers
  or refusals. The workflow is unaffected (it never reaches 80%).
- **Groq `output_parse_failed` (`rr/llm.py`):** retried once with a format
  reminder, like the three other model-caused 400s. It cost agent mh07 and
  mh09 on 14 October.
- **Rate limits (`eval/run_compare.py`):** a record whose error is a
  `TransientLLMError` (a 429, 5xx or network failure that outlasted the
  backoff) goes to the dead letter and is retried on resume, instead of being
  scored as the system's failure. live1.jsonl held 27 agent and 23 workflow
  such records. It changes what an `error` record means: from now on it is a
  model or request failure, never a quota.
- **Fence (`rr/fence.py`):** zero-width spaces, soft hyphens and word joiners
  are removed before tag and marker detection, so `<\u200b/untrusted_source>`
  no longer closes the fence unseen. `injection_flags` can only go up.
- **Results API (`api/results.py`):** `*.judge.jsonl` and `dead_letter.jsonl`
  are no longer listed or scored as runs.
- **Not changed, open questions:** the workflow's `k=4` against the agent's
  self-chosen `k` up to 8, the single calculator expression, and McNemar
  pairing on (question, repeat) when repeats are cached replays. Each changes
  the comparison itself, so each is a design decision, not a fix.

## 2026-11-06: Design changes from the audit (plan, calculator, last call, pairing, guards)

- **Workflow plan:** each search names its own corpus (1 to 4 searches)
  instead of every query going to every corpus. Queries are written in
  English in both systems (`QUERY_LANGUAGE_RULE`); the agent's prompt still
  said "search the French corpus in French", left over from the French primer
  replaced on 2026-10-11.
- **Calculator:** the workflow asks for 1 to 3 expressions in its one calc
  call, and the shared calculator gains `days_between` for ISO dates. Both
  systems get the date function, so it does not favour one.
- **Agent last call:** built from the fenced passages, the calculator results
  and the question, like the workflow's answer step, instead of the tool-call
  conversation sent without tools (gpt-oss called a tool anyway and Groq
  rejected the request twice, agent cc03).
- **Retrieval depth, not equalised:** the workflow keeps k=4, the agent picks
  up to 8 (it chose 8 on 68 of 87 searches). At k=8 the workflow's 4 searches
  would overrun the 12k-token budget and Groq's 8k-token request limit.
  `RR_WORKFLOW_SEARCH_K` lets a run test it. The difference is now stated in
  the deep dive instead of being hidden behind "same tools".
- **McNemar:** pairs are questions (strict majority over repeats), not
  (question, repeat). With the LLM cache on, repeats are replays and the old
  pairing made p-values far too small (5 discordant questions over 3 repeats:
  p = 6e-5 instead of 0.0625). On the published one-repeat run nothing
  changes but the sentence.
- **What changes in the numbers:** the plan, calculator, prompt and last-call
  changes alter both systems' behaviour, so runs before this date are not
  comparable with runs after it. The published figures are still the
  14 October records; a new run is needed to measure the effect.
- **API guards:** `RR_API_KEY` and `RR_RATE_LIMIT_PER_MINUTE` protect
  `/api/ask`, both off by default; `render.yaml` sets the rate limit to 6.
- **Stack:** the Dockerfile builds grounded-rag at the same pinned commit as
  CI (68b8b93, checked by a test; `eval.locate_gold` reports nothing missing),
  instead of `main`. The image installs `requirements.lock` (Python 3.12,
  Linux), test tools moved to `requirements-dev.txt`, `npm ci` no longer falls
  back to `npm install`, CI builds the UI on Node 26 like the image, Actions
  are pinned to SHAs, CI runs on pushes to main and on PRs (not twice per PR
  push), the compose Ollama image is pinned, and Dependabot may now propose
  Python 3.13 and 3.14 base images (only 3.15 lacks a zstandard wheel).


## 2026-11-08: grounded-rag pin moved to e378805

- **What:** `GROUNDED_RAG_REF` in the Dockerfile and in `.github/workflows/ci.yml`
  goes from 68b8b93 to e378805, grounded-rag's main after #33.
- **Why:** grounded-rag's main history was rewritten on 8 November to drop
  commit trailers. The code is identical (68b8b93 and its rewrite 63918be have
  the same tree), but 68b8b93 is no longer on any branch, so CI and the image
  build would fail once GitHub prunes it. Moving to e378805 also brings #33:
  "(S9)" counts as a citation.
- **What changes in the numbers:** citation counts can only go up (one stored
  agent answer, sh08, cites in parentheses). `eval.locate_gold` reports nothing
  missing at e378805; 262 tests pass against it.

## 2026-11-08: Second audit, six live fixes

- **Stored rate-limit errors are left out (`eval/report.py`, `api/results.py`):**
  `run_compare` stopped writing `TransientLLMError` records on 6 November, but
  `live1.jsonl` and `live2.jsonl` still hold 50 and 7 of them, and the report
  and the Results tab scored each as a failure. They are now counted in
  `infra_errors`, left out of every rate and average, and the count is printed.
  - **What changes in the numbers:** live1 goes from workflow 2/32 and agent
    2/32 (6.2% each) to 2/9 and 2/5; its average LLM calls stop counting
    zero-call quota records. The published `r120e` report is byte-identical.
- **Cached repeats (`eval/report.py`):** with the LLM cache on, repeat 1 and
  later replay repeat 0, so the spread across repeats was 0.0 by construction
  and printed as "± 0.0". It is now n/a, with a note, and the Wilson interval
  counts questions (strict majority over repeats, as McNemar already does)
  instead of records. For 40 questions × 3 cached repeats at 75% the interval
  widens from 67-82% to 60-86%. Cache-off runs and one-repeat runs are unchanged.
- **Loop stops are tagged `loop_or_budget` (`rr/common.py`, `eval/score.py`):**
  a run the guard ends for repeating a call has `budget_exhausted` false, so it
  fell through to `missed_hop` or `uncited`. Records now carry `stop_reason`,
  and the tag also applies when it is `loop`. Pass/fail is unchanged; only the
  failure tag moves. No stored record has a loop stop.
- **Judge estimate (`eval/judge.py`):** errored records were listed as pending,
  so `--estimate` counted them as LLM calls (80 instead of 76 on `r120e`,
  64 instead of 10 on `live1`) and every resume logged them as judge errors
  again. They are no longer pending. Verdicts and judged rates are unchanged.
- **Calculator output (`rr/tools.py::format_result`):** results were printed
  with `:g`, six significant digits with e-notation from a million up
  (`416161 * 3 = 1.24848e+06`), next to a prompt calling them exact. Whole
  numbers are now printed in full and fractions with 12 significant digits.
  This changes the prompts of calculator questions, so cached replies for
  them no longer match; no stored record held a value over a million.
- **Agent last call (`rr/agent.py`):** a calculate call rejected for invalid
  arguments was handed to the last call as "Calculator result (exact): Invalid
  arguments: ...". Only calls that ran are passed on now.

## 2026-11-08: Two regressions from 2026-11-06 found by a re-run and fixed; results re-run

- **What:** grounded-rag is pinned to e9535b2 (its main after #34) in the
  Dockerfile and CI. The first run on it, `r121e`, showed two regressions
  from the 6 November design changes, both fixed and measured again in
  `r122e`, which the README and `docs/results.md` now report.
- **Workflow plan:** with a minimum of 1 search, the paired plan asked for a
  single search on 26 of 40 questions (mean 1.35 searches against 2.12 on
  14 October). Source recall fell from 0.85 to 0.73 and the workflow refused
  8 answerable questions (27/40). `MIN_SEARCHES = 2` and a prompt that asks
  for one search per part of the question bring it back to a mean of 3.83
  searches, recall 0.83, 2 false refusals, 30/40.
- **Agent budget:** `nearly_exhausted` assumed the next call no larger than
  the last; agent calls grow by about 1k tokens per k=10 search, so 8 runs in
  `r121e` reached ~12,100 tokens after 4 calls and stopped on the budget
  message. It now adds the growth seen between the last two calls. In
  `r122e` one run stops that way (cc02); three more hit the budget during the
  verification retry, after a checked answer that is kept.
- **What changes in the numbers:** `r122e` gives workflow 30/40, agent 30/40,
  both pass 28, McNemar p = 1.000. The workflow now spends 4,094 input tokens
  a question (2,841 on 14 October) because it searches more. Runs before this
  date are not comparable with `r122e`: the plan, the budget guard and
  grounded-rag's verifier (#34, a citation after the full stop now cites the
  sentence before it) all changed.

