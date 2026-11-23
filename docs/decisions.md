# Decisions

One entry per decision that still holds: date, what, why, in a few lines. An
entry that replaces an earlier one is merged into it, with the earlier date in
a few words. Entries that explain what a number means stay, even when the run
behind them is old.

## 2026-10-10: Approving over the API needs an identity

- **What:** `POST /api/approve` checks who calls it. With `RR_APPROVE_TOKEN`
  unset only this machine may approve (403 otherwise); set, the
  `X-Approve-Token` header must match, compared in constant time (401). The
  response carries `decided_by`. The three other routes only read.
- **Why:** approving is the agent's one irreversible action, and with
  `--host 0.0.0.0` anyone on the network could write a report in the owner's
  name. A token is one shared secret, not a user: with several users, replace
  it with a session. A UI deployed off-box must send the header
  (`web/src/AskTab.jsx`).

## 2026-10-10: One Docker image, grounded-rag cloned inside

- **What:** `Dockerfile` (Node stage for the UI, Python runtime with
  grounded-rag cloned at a pinned ref), `docker/entrypoint.sh` (optional index
  build at start, uvicorn on `$PORT`), `docker-compose.yml`, `render.yaml`.
- **Why:** route-or-roam reaches grounded-rag through `sys.path` and uses its
  corpora and indexes, so a checkout at a known ref is the deployment, not a pip
  install. The Render blueprint is written from the platform's docs, not from a
  deploy: the first deploy is its test. Rebuilding the indexes on every start
  costs embedding calls, fine for two small corpora.

## 2026-10-11: Eval set relabelled against the full AI Act

- **What:** grounded-rag replaced its four-file French AI Act primer with the
  full Regulation (EU) 2024/1689 in English (recitals, articles, annexes). The
  15 questions on the AI Act were relabelled in `eval/questions.jsonl`. Label
  only: sh07, sh09, mh07, cc01, cc03, cc04, cc06. Annex III date (mh08, cc02,
  in03): 2 December 2027 becomes **2 August 2026**, since the later date came
  from the proposed Digital Omnibus, which is not in the corpus. sh10 (why
  Annex III was postponed) is now **unanswerable**; un03 (maximum fine) is now
  **answerable** (Article 99: EUR 35 000 000 or 7 %). mh09 and cc05 ask for the
  date GPAI obligations apply (2 August 2025) and prohibited practices were
  banned (2 February 2025); sh08 asks for the prohibited and high-risk
  categories; mh08 drops "explainability" (a recital only).
- **Why:** every AI Act gold source pointed at a file that no longer exists, so
  source recall read 0 and correct answers would have scored as wrong.
- **Numbers:** pass rates, source recall and refusal counts from before this
  date are **not comparable** with later runs. `python -m eval.locate_gold`
  lists every gold source missing from the grounded-rag checkout and where its
  facts now live (exit 1 while any is missing); it reports, never relabels.

## 2026-10-12: An opt-in LLM judge, next to the string score

- **What:** `eval/judge.py` makes one structured call per stored record and
  writes verdicts to their own jsonl. `eval.report --judge` adds rows
  ("success with judged facts", and where judge and strings disagree).
- **Why:** string matching rejects answers that are right in other words
  (`missed_hop`). The judge lists them without making the reference score depend
  on a model: without `--judge` the report is byte-identical, and "success" is
  still `eval/score.py`.
- **Open:** the judge has not been run on the published records, so its
  agreement with a human reader is unmeasured. Read the "judge yes, strings no"
  list by eye before quoting a judged rate.

## 2026-10-13: The fence catches spaced, fullwidth and zero-width tags

- **What:** `rr/fence.py` matches `<untrusted_source` look-alikes on an NFKC
  copy of the passage, with whitespace allowed around the slash, after removing
  zero-width spaces, soft hyphens and word joiners (added 2026-11-06). It
  escapes only the matched spans, and the source name is attribute-escaped.
- **Why:** `< /untrusted_source>`, the fullwidth form and a zero-width space
  inside the tag let a passage close its own fence, and a source name with a
  quote could add attributes to the tag. Same fix as grounded-rag's.
- **Numbers:** `injection_flags` also counts these spellings, so it can only go
  up on the same corpus; runs from before are not strictly comparable on it.

## 2026-11-06: Scoring and refusal rules

- **Labels:** sh08 accepts "interdits/interdite/prohibé(e)(s)", mh07
  "subliminale(s)/subliminaux", mh09 "fournisseur". The questions are in French
  and a right answer failed for a missing inflection. Re-scoring the 14 October
  records (no LLM call): workflow 28/40 to 30/40, agent 29/40, no record went
  from pass to fail.
- **Refusal (`rr/common.py::is_refusal`):** an answer is a refusal only when it
  is the refusal sentence and nothing else (curly apostrophes folded first).
  Before, "Revenue was 416 billion. I couldn't find anything..." passed an
  unanswerable question. On the 254 stored records no `refused` value changed.
- **McNemar (`eval/report.py`):** pairs are questions (strict majority over
  repeats), not (question, repeat). With the cache on, repeats are replays and
  the old pairing made p far too small (5 discordant questions over 3 repeats:
  6e-5 instead of 0.0625).
- **Rate limits (`eval/run_compare.py`):** a `TransientLLMError` (a 429, 5xx or
  network failure that outlasted the backoff) goes to the dead letter and is
  retried on resume. An `error` record now means a model or request failure,
  never a quota (live1.jsonl held 27 agent and 23 workflow such records).
- **Groq `output_parse_failed`:** retried once with a format reminder, like the
  three other model-caused 400s. It cost agent mh07 and mh09 on 14 October.

## 2026-11-06: Design changes (plan, calculator, last call, search depth, guards)

- **Workflow plan:** each search names its own corpus, 2 to 4 searches
  (`MIN_SEARCHES = 2`, set on 8 November: with a minimum of 1, 26 of 40
  questions got one search, source recall fell from 0.85 to 0.73 and the
  workflow refused 8 answerable questions, 27/40; with 2 the mean is 3.83
  searches, recall 0.83, 30/40). Queries are written in English in both systems.
- **Calculator:** the workflow asks for 1 to 3 expressions; both systems get
  `days_between` for ISO dates. Results print in full (whole numbers) or with 12
  significant digits, not `:g` (`416161 * 3 = 1.24848e+06` sat next to a prompt
  calling them exact).
- **Agent last call:** built from the fenced passages, the calculator results
  that ran (a rejected call is left out) and the question, like the workflow's
  answer step. `nearly_exhausted` is true with one step left, at 80% of the
  tokens, or when one more call as big as the last, plus the growth seen between
  the last two, would reach the cap. The agent's prompt grows by about 850
  tokens per k=8 search (fit on 98 agent runs of r120e to r123e, 95% bootstrap
  interval 730 to 960). Before this guard, 8 runs of `r121e` went past the cap
  (12,100 to 16,139 tokens after 4 or 5 calls), 5 of them with no answer.
- **Search depth, not equalised:** the workflow keeps k=4, the agent picks up to
  8 (it asked for 10, capped to 8 in code, on 68 of 87 searches).
  `RR_WORKFLOW_SEARCH_K` lets a run test it; at k=8 the workflow's searches
  would overrun the 12k budget and Groq's 8k request limit.
- **API guards:** `RR_API_KEY` and `RR_RATE_LIMIT_PER_MINUTE` protect
  `/api/ask`, both off by default; `render.yaml` sets the limit to 6.
- **Numbers:** these change both systems' behaviour, so runs before `r122e`
  are not comparable with it (it also has grounded-rag's #34: a citation after
  the full stop cites the sentence before it).

## 2026-11-08: Report rules for stored runs

- **Rate-limit errors left out (`eval/report.py`, `api/results.py`):**
  `live1.jsonl` and `live2.jsonl` hold 50 and 7 `TransientLLMError` records.
  They are counted in `infra_errors` and left out of every rate: live1 goes
  from 2/32 for both systems to workflow 2/9 and agent 2/5.
- **Cached repeats:** repeats after the first replay it, so their spread is
  zero by construction. It prints as n/a, and the Wilson interval counts
  questions (majority over repeats) instead of records: for 40 questions at 75%
  it widens from 67-82% to 60-86%. Cache-off runs are unchanged.
- **Loop stops** are tagged `loop_or_budget` (records carry `stop_reason`);
  pass/fail is unchanged.
- **Judge `--estimate`** no longer counts errored records as pending calls.

## 2026-11-16: grounded-rag pin is 2fad55f

- **What:** the Dockerfile and CI pin grounded-rag at 2fad55f, up from 68b8b93
  (6 November), e378805 and e9535b2 (8 November). `eval.locate_gold` finds every
  gold source at the pin (314 corpus files, 0 missing).
- **Why:** grounded-rag's main history was rewritten on 8 November, and the
  old pin is on no branch any more, so CI and the build would fail once GitHub
  prunes it. A pin also keeps an upstream corpus change from turning CI red on
  its own. To bump: check out the new commit, run `eval.locate_gold`, relabel,
  change both shas (a test checks they agree).
- **Retrieval at 2fad55f:** hybrid retrieval changed in 43cde89 (a BM25-only
  passage carries its dense cosine; MMR's relevance is the fused rank). The
  source files a workflow search returns differ in 32 of 40 records between
  e9535b2 and 2fad55f, but replaying the same recorded queries moves source
  recall by at most 0.01. The workflow's recall drop (0.83 to 0.80) came from
  the planner writing other queries. `r126e` (2026-11-22) later tested this
  with LLM runs.
- **Interval:** `eval/report.py` computes the Wilson interval over all records
  when the cache is off. Three repeats of 40 questions are not 120 independent
  draws, so the README also gives it on the 40 questions. Whether the report
  should count questions is open.

## 2026-11-16: Calculator results and the question's own numbers pass the figure check

- **What:** `rr/common.py::check_answer` runs grounded-rag's verifier, then
  accepts two kinds of figure its figure check rejects: the result of a
  calculation that ran (`calculator_values`, matched at the precision the answer
  writes it, a one-digit figure only when equal) and a number the question
  gives (`grounded.figures_in`). Any other figure absent from the cited source
  still fails; the rejected ones are in `verify["unsupported_numbers"]` and the
  retry message names them. Both systems pass it the question and their
  tool calls (pinned in `tests/test_derived_figures.py`).
- **Why:** grounded-rag has no calculator, so a correct total is "invented" to
  it. On the agent's stored `r123e` states, 35 of 48 retried first answers
  failed on a figure: 19 were calculator results, 6 came from the question.
  The old retry message offered the refusal sentence and the workflow sometimes
  took it (nu05, nu08, cc06).
- **Checked:** on the same 48 answers, with no LLM call, the rule clears 18 of
  the 35 and 15 pass on the first try (0 before). What still fails is
  arithmetic done without the calculator and identifiers the passage doesn't
  repeat ("10-K").
- **Numbers:** both systems now verify numeric and cross-corpus answers
  differently, so runs before this entry are not comparable with `r124e` on.
  `eval.report --out runs/x.md` now writes its chart next to the report; it used
  to overwrite `docs/success_by_type.png`.

## 2026-11-20: Results are `r125e`, run on one machine

- **What:** the README, `docs/results.md` and the deep dive report `r125e`: 40
  questions × 2 systems × 3 repeats, `openai/gpt-oss-120b` on Groq, cache off,
  grounded-rag at 80ef892, with the figure-check fix. All 240 records ran in one
  cloud sandbox, 16 to 19 November.
- **Why not `r124e`:** it stopped at 19/240 on the laptop. The laptop's and the
  sandbox's Ollama can return slightly different embeddings, so finishing it in
  the cloud would have mixed two embedders. It was rerun whole.
- **Numbers:** workflow 85/120, agent 99/120; on the 40 questions 29 both, 4
  agent only, 7 neither, McNemar p = 0.125. The agent stopped on the budget
  message in 8 of 120 runs; one agent record (mh08, repeat 1) is a Groq HTTP
  400, scored as a failure.
- **What changed in the story:** the split is the same as `r123e` (0 and 4) but
  not the questions. mh03 and nu08 now pass for the workflow; its other wrong
  refusals are retrieval misses (14 of 15, from replaying the recorded
  searches, which retrieval being deterministic allows). "The refusal is the
  model's, not the gate's" and "the two systems are level" (`r122e`) were wrong.
- **Rate-limit reasons:** when every key returns 429, the error keeps Groq's
  reason per key (`rr/llm.py::_limit_reason`: limit kind, used over limit,
  tokens requested, organisation prefix, never the key). The message still
  starts with "rate-limited on every configured key (429)". Added because the
  daily cap (TPD) refuses a 6,000-token call while a 700-token one still fits.

## 2026-11-22: `r126e` control run: the move from `r122e` was mostly chance

- **What:** `r126e` is `r125e`'s route-or-roam code on grounded-rag e9535b2,
  the commit `r122e` used (`rag/answer.py` and `rag/verify.py` are identical at
  both, only retrieval differs, 43cde89). 240/240 records, same cloud machine,
  sha256 of the records file equal on both sides.
- **Numbers:** workflow 88/120 (31, 30, 27 per repeat), 30/40 on the majority
  (Wilson 60-86%); agent 95/120 (33, 32, 30), 32/40 (65-90%). Only the workflow
  1 (nu01), only the agent 3 (sh09, mh02, cc06), both 29, neither 7; McNemar
  p = 0.625. False refusals: 15 for the workflow, 5 for the agent. The agent
  stopped on the budget message in 12 of 120 runs and 28 runs ended above 12,000
  tokens (23 in `r125e`).
- **Conclusion:** `r122e` and `r125e` differ at the majority on 8 question ×
  system pairs. `r126e` sides with `r125e` on 6 and with `r122e` on 2 (workflow
  mh07, agent nu01). So chance explains most of the move, and 2 pairs can't size
  a retrieval effect. Reproduce it with `python -m eval.compare_runs
  runs/r122e--groq-openai-gpt-oss-120b.jsonl runs/r125e--groq-openai-gpt-oss-120b.jsonl
  runs/r126e--groq-openai-gpt-oss-120b.jsonl` (`r122e` is not in the repository:
  its file stays in the local `runs/`).
- **README claims corrected:** cross-corpus questions are 6 of the agent's 14
  extra passes in `r125e` and 2 of 7 in `r126e`, so "most of the gap" was wrong;
  the agent answers sh10 (unanswerable) once in `r126e` (8/9); 0/6 questions are
  4 in `r125e` and 7 in `r126e` (cc03, cc05, in03 added); with the old retrieval
  the workflow still refuses 15 answerable questions, on another set, 12 of
  them missing a gold source the answer needed (replayed on 23 November, see
  below); the budget message ended 8 agent runs (12 in `r126e`).

## 2026-11-22: Docs cut down, runs made traceable

- **What:** the README keeps what the project is, results, limits and commands;
  CI, Docker, Render and Streamlit moved to `docs/deploy.md`; old runs
  (`r120e` to `r124e`, `s120*`) are summarised in the deep dive. The records and
  reports of `r125e` and `r126e` are un-ignored in `.gitignore`, so the
  README's figures can be recomputed from the repository.
- **Why:** short docs and one decision log. A figure nobody can recompute is a
  claim, not a result.
- **Tests:** 287 to 291: the agent's call to `check_answer` (question and tool
  calls) is pinned, and so is the private `rag.verify._numbers` that
  `rr/grounded.py` imports; `eval/compare_runs.py` has two offline tests.

## 2026-11-23: grounded-rag pin moves to 3b09461

- **What:** the Dockerfile and CI pin grounded-rag at 3b09461 (its `main` after
  the 22 November audit), up from 2fad55f. Published runs keep their commit:
  `r125e` on 80ef892, `r126e` on e9535b2. New runs use 3b09461.
- **Why:** the pin had fallen behind grounded-rag's fixes. Retrieval is the same
  as 80ef892 and the system prompt is unchanged; the verifier now reads "n't"
  negations and thousands separators correctly, which can change which figures
  `check_answer` flags. route-or-roam passes its own grounding bar, so
  grounded-rag's new default (0.30) does not apply.
- **Replay of the workflow's false refusals,** same index, recorded queries:
  `r126e` on e9535b2, 12 of 15 miss a gold source the answer needed (only sh08
  got none at all); `r125e` on 80ef892, 14 of 15. No LLM call.
