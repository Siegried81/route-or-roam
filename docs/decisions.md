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
  runs/r126e--groq-openai-gpt-oss-120b.jsonl` (`r122e`'s records and report are
  in the repository too).
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

## 2026-11-23: Search depth cap made configurable, for a k=4 agent ablation

- **What:** the most passages one search returns is `RR_MAX_SEARCH_K`
  (`settings.MAX_SEARCH_K`, default 8, the old hard-coded value). Run records
  store `max_search_k` and `workflow_search_k`.
- **Why:** in `r125e`, 209 of the agent's 238 searches asked for k=10 and got 8;
  the workflow always searches at 4. So the agent's lead mixes the loop with
  twice the depth. `RR_MAX_SEARCH_K=4` separates them. Raising `RR_MAX_TOKENS`
  instead would break the shared budget.
- **State:** nothing published changes. The ablation is not run yet (about 1M
  Groq tokens for 120 agent records). If the lead holds, the loop explains it;
  if it vanishes, retrieval depth does and the README must say so.

## 2026-11-23: Runs record their code, and cost is per correct answer

- **What:** every new record carries `code_sha`, `code_dirty` (`runs/` and
  `reports/` ignored, since a run writes there), `grounded_rag_sha` (`unknown`
  outside a git checkout) and `questions_sha256`. `eval/cost.py` gives tokens
  and dollars per correct answer: mean tokens in plus out per record over the
  pass rate. `eval/check_run.py` joins the repo.
- **Why:** `r125e` and `r126e` differ by the grounded-rag commit, and nothing in
  their files says so; the README had to. Failed runs spend tokens too, so cost
  is read per answer that passes. Groq's list price ($0.15 in, $0.60 out per
  million) comes from an undated page read on 2026-11-23; recheck it before
  quoting dollars. Existing runs files are not rewritten and have no such fields.
- **Tests:** 292 to 301 (provenance, check_run, cost). The README now opens with
  the question, the decision rule and the cost; the run-by-run reading moved to
  the deep dive.

## 2026-11-23: grounded-rag pin moves to 3b2de9f

- **What:** the Dockerfile and CI pin grounded-rag at 3b2de9f (its `main`), up
  from 3b09461. `r125e` (80ef892) and `r126e` (e9535b2) keep their commit, and
  neither was run on the pin.
- **Why:** the pin had fallen behind `main`. Between the two commits grounded-rag
  adds a `filings_statements` corpus (Item 8 of the 10-K), which route-or-roam
  does not list in `CORPORA`; a chunker by lines for files that start with
  `Context:`; Azure backends; and a `.env` lookup limited to its own folder. A
  bare `load_dotenv()` used to walk up the folder tree.
- **Checked, no LLM call:** `data/filings_sections` and `data/ai_act_sections`
  are byte-identical at both commits; no file of either starts with `Context:`;
  both corpora chunk to the same 155 and 878 chunks with the same ids and text
  (hash compared); `rag/retrieve.py` and `rag/verify.py` are untouched.
  `python -m eval.locate_gold` finds 314 corpus files and 0 missing gold sources,
  and the 40 questions point to 59 sources, all present.
- **Bug fixed, `grounded_rag_sha`:** `git -C <path> rev-parse HEAD` walks up the
  tree, so a grounded-rag folder that is not a checkout but sits inside another
  repository (the archive RR_FETCH_GROUNDED_RAG unpacks under route-or-roam)
  recorded route-or-roam's own commit. Now it is `unknown` unless the folder is
  the root of a checkout. Records also carry `grounded_rag_dirty`.
- **Docs corrected:** the 2026-11-22 entry said `r122e` is not in the
  repository, but it is tracked; "15 answerable questions" refused were 15 records (the
  README, deep dive section 9.1); the cross-corpus lead is 10/18 against 4/18 in
  `r125e` but 8/18 against 6/18 in `r126e`, which the README had left out; the
  k=10 failure of `s120b` was 7 of the agent's 8 failures, not "7 of 17"; the
  eval doc now says to set `RR_LLM_CACHE=0` and `--models` for the published
  file names.
- **Tests:** 301 to 307 (grounded-rag provenance, `eval/cost.py` edge cases,
  `check_run --provider`, `RR_MAX_SEARCH_K=0`).

## 2026-11-23: Numbers are compared at the precision the answer shows

- **What:** a number of the answer matches a numeric key fact when the fact,
  rounded (half up, on the decimal value) to the decimals and the scale written
  in the answer, equals the number written. This replaces the ±1% band of
  `eval/score.py`. "8,0 billion" matches 7,979 (millions) and "6 %" matches 6.4;
  "15 413" does not match 15 421, nor does "6.5" or "6.42" match 6.43. A scale
  word or suffix (million, billion, B, bn, Mrd, miljard and others) is applied,
  and a fact of 1000 or more may be read as units, thousands or millions. A
  number is always also tried as written, because a fact's unit is not known.
  The LLM judge's rubric says the same (new `RUBRIC_ID`).
- **`min_decimals`:** a question may carry this optional integer, the precision
  its text asks for. The number must then also show at least that many
  decimals, at the fact's own scale; a fact never has to show more decimals than
  its label has (an integer fact needs none). Set to 1 on `nu01`, `nu02`, `nu03`
  and `nu07`, the four questions that say "one decimal". The precision rule alone
  accepted "6 %" for the 6.43% of `nu01`, which does not answer "one decimal".
- **Label corrected, `nu03`:** 3.825 becomes 3.846. Source `05_mdna.txt`,
  "Greater China 64,377 (4) % 66,952": 1 - 64,377 / 66,952 = 0.038460 = 3.846%.
  3.825 was a midpoint chosen so that the old ±1% band accepted 3.8 and 3.85;
  with the precision rule the true value does it directly (3.8 and 3.85 pass,
  3.9 and 3.84 fail). Nothing else in `eval/questions.jsonl` changed.
- **Why:** a ±1% band is the wrong test of "same number". It accepted 15 413 for
  15 421 (shown before the fix) and rejected a rounding to the digits the answer
  claims. Reading the answer at its own precision accepts honest rounding and
  rejects a wrong digit; `min_decimals` keeps a question's own demand.
- **Numbers:** re-scored offline from the stored answers (no LLM call) with
  three rules: ±1%, precision alone, precision plus `min_decimals`. Precision
  alone moved 3 records from fail to pass (`nu01` answered "6 %": `r122e` agent,
  `r126e` agent repeats 1 and 2) and none the other way; with `min_decimals`
  those 3 fail again, so the final rule gives the same verdict as ±1% on all 560
  records (checked again after the label corrections). Every total, majority, McNemar p, count by type and cost per correct
  answer is unchanged. Earlier entries quote the figures of their day.
- **Labels corrected, `nu01`, `nu02`, `nu07`:** 6.4 becomes 6.43, 26.2 becomes
  26.23, 9.7 becomes 9.67 (`eval/paraphrases.jsonl` follows for `nu02` and its
  two rephrasings, which also get `min_decimals: 1`). They were rounded to the
  one decimal the question asks for, so the full "6.43 %" failed against 6.4.
  From the sources in `05_mdna.txt` and `06_market_risk.txt`: 416,161 / 391,035
  - 1 = 0.064255, 109,158 / 416,161 = 0.262298, 590 / 538 - 1 = 0.096654. With
  `min_decimals: 1`, "6.4 %" and "6.43 %" both pass for 6.43 (6.43 rounded to
  one decimal is 6.4), "6 %" and "6.5 %" fail; likewise for the other two. The
  other numeric facts were checked and are exact: 91.3 and the integers are
  quoted from the source, 41616.1 is 10% of 416,161, 104.7 is 89.3 + 15.4.
- **Limit:** a one-digit answer is accepted over a wide range when a question sets
  no `min_decimals`.

## 2026-11-23: A refusal that cites its sources is still a refusal

- **What:** `rr/refusal.py::detect_refusal`, used by `rr/common.py::is_refusal`
  at run time and by `eval/score.py` when scoring (the recorded `refused` flag
  OR the answer text). An answer is a refusal when every sentence is a refusal,
  or when its first sentence is a strong refusal and the rest only describes what
  the sources contain, with or without `[S#]` markers. A hesitation followed by a
  claim ("I'm not sure, but revenue was 416 billion [S1].") is an answer, as is a
  claim before the refusal sentence. `cited_refusal` is in the result record and
  in the score. Before, only grounded-rag's exact sentence counted, and only
  without a valid citation.
- **Languages:** strong phrases in English, French, Dutch, Spanish and German,
  tested one by one. The run's own stop messages ("I stopped before finishing")
  are not refusals. The semantics follow grounded-rag's `answer_metrics.is_refusal`
  (strong phrase, negation about the sources, first sentence); the code is
  written here and does not import it, because the pin is not that branch.
  A test checks that every exact refusal sentence of the pinned grounded-rag is
  detected.
- **Numbers:** on the 560 stored records of `r122e`, `r125e` and `r126e` no
  `refused` value changes and none is a cited refusal, so no verdict moves for
  this reason.
- **Limit:** "I don't know. The documents say revenue was 416 billion [S1]."
  counts as a refusal, as decided: a strong refusal followed by a description of
  the sources. On a real answer set it would be worth reading these by eye.
- **Tests:** 307 to 376 (number rule, `min_decimals`, refusal rule in five
  languages, the hesitation-then-claim guard, `cited_refusal`).
