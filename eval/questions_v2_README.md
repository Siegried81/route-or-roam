# questions_v2.jsonl: 60 held-out questions

**How it was written.** On 2026-11-23, by an AI coding assistant (Claude Code agent) working for the
project owner, straight from the corpus files in grounded-rag (`ai_act_sections`, `filings_sections`).
No `runs/` file, no system answer and no LLM call was used; the quoted sentences in each `note` were
checked mechanically against the files, and each unanswerable question was grep-checked against the
corpus. The set is held out: do not tune prompts, aliases or retrieval on it, or it stops being one.

**Mix (60).** single_hop 12, multi_hop 12, numeric 10, cross_corpus 14, unanswerable 6, injection 6.
AI Act questions are in French, as in `questions.jsonl`; 10-K ones in English. About 39 of 60 touch the
AI Act (25 only, 14 crossed), 35 touch the 10-K (21 only). Ids continue the v1 numbering
(`sh11`..`sh22`, `mh11`..`mh22`, `nu09`..`nu18`, `cc07`..`cc20`, `un04`..`un09`, `in04`..`in09`), so
v1 + v2 can be concatenated into 100 questions with unique ids.

**Format.** Same fields as `questions.jsonl`, plus an optional `min_decimals` (integer): when present, a
number in the answer only counts if it shows at least that many decimals ("6 %" is refused for a fact
asked "to one decimal"). It is set on every question whose wording asks for a precision
(`one decimal` gives 1, `two decimals` gives 2). Numeric facts are stored at full precision, as
printed in the source or computed exactly from it (a ratio keeps six decimals), so a more precise
answer is not penalised. `note` holds the exact source sentences, and for computed answers the
arithmetic after `computed:`.

**Review (2026-11-23).** An independent review rewrote or corrected 31 questions before the run: the
cross-corpus questions that only juxtaposed two unrelated facts (cc12 to cc20, mh20) now link the
two corpora by subject; four multi-hop questions that one article could answer alone (mh11, mh14 to
mh17, mh21) now need both sources; weak facts (cc17 "yes" and "60", mh22) were replaced; alias gaps
(sh13, sh15, nu11, cc09) were filled.

**Use.**
```
python -m eval.locate_gold --questions eval/questions_v2.jsonl        # 0 missing gold source
python -m eval.run_compare --run-id v2a --questions eval/questions_v2.jsonl --repeats 2 --estimate
python -m eval.check_run runs/v2a.jsonl --questions eval/questions_v2.jsonl --repeats 2
python -m eval.report --runs runs/v2a.jsonl --questions eval/questions_v2.jsonl
python -m eval.compare_runs runs/a.jsonl runs/b.jsonl --questions eval/questions_v2.jsonl
```
`RR_QUESTIONS=eval/questions_v2.jsonl` does the same for every tool; `--questions` wins over it. For
the 100-question set, `cat eval/questions.jsonl eval/questions_v2.jsonl > eval/questions_all.jsonl`
and pass that file. Each record's `questions_sha256` is the hash of the file actually read.

`eval/questions_all.jsonl` is `questions.jsonl` followed by `questions_v2.jsonl`
(100 questions, ids unique). Rebuild it with
`cat eval/questions.jsonl eval/questions_v2.jsonl > eval/questions_all.jsonl`.
