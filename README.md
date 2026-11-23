# Route or Roam

[![CI](https://github.com/Siegried81/route-or-roam/actions/workflows/ci.yml/badge.svg)](https://github.com/Siegried81/route-or-roam/actions/workflows/ci.yml)

**Same questions, same tools, same model: when is an agent worth it?**

On 40 grounded questions (Apple's FY2025 10-K and the EU AI Act), the agent
passes more than the workflow in both runs, 99 against 85 records of 120 and
then 95 against 88. The gap is not statistically significant, and a bad search
caps both systems. The agent uses about 1.5 times the tokens per correct answer
but the same dollars (about $0.002), and its slowest runs are slower: p95
latency 25 to 29 s against 9 to 13 s.

[grounded-rag](https://github.com/Siegried81/grounded-rag) is the RAG engine,
route-or-roam is the test bench. Retrieval, the answer prompt and the verifier
come from grounded-rag, imported in place, never copied. CI and the Docker
image pin it at commit 3b2de9f; the two published runs used older commits (see
[Limits](#limits)).

## Workflow or agent

- **Workflow** (`rr/workflow.py`): code fixes the route, at most 4 LLM calls. Pick it when cost and latency must be bounded, or when the process has to be strict and auditable.
- **Agent** (`rr/agent.py`): a ReAct agent that chooses its tools and how often. Pick it when questions cross several sources: cross-corpus questions passed 10/18 against 4/18 in `r125e`, but 8/18 against 6/18 in `r126e`.
- **Either way the search is the ceiling.** In `r125e` 15 of the workflow's 120 records were wrong refusals of answerable questions, and in 14 of them a source the answer needed never came back from retrieval (12 of 15 in `r126e`). Fix retrieval before choosing a design.
- In dollars the two cost about the same per correct answer: output tokens cost four times the input, and the workflow writes about twice the output of the agent.

## What a correct answer costs

Tokens in plus out per record, divided by the pass rate (`python -m eval.cost`).
Dollars use Groq's `openai/gpt-oss-120b` list price, $0.15 per million input
tokens and $0.60 per million output, read on
[console.groq.com](https://console.groq.com/docs/model/openai/gpt-oss-120b) on
23 November 2026 (the page carries no date).

| run | system | pass rate | tokens per record | tokens per correct answer | dollars per correct answer |
|---|---|---|---|---|---|
| `r125e` | workflow | 85/120 | 3585 + 1355 | 6,974 | $0.0019 |
| `r125e` | agent | 99/120 | 7674 + 683 | 10,130 | $0.0019 |
| `r126e` | workflow | 88/120 | 3436 + 1287 | 6,440 | $0.0018 |
| `r126e` | agent | 95/120 | 7528 + 691 | 10,383 | $0.0020 |

## Where it stands

40 questions × 2 systems × 3 repeats, `openai/gpt-oss-120b` on Groq, LLM cache
off. `r125e` ran on grounded-rag 80ef892 ([report](docs/results.md)), `r126e`
on e9535b2 ([report](runs/r126e.md)), each on one cloud machine.

| | workflow `r125e` | agent `r125e` | workflow `r126e` | agent `r126e` |
|---|---|---|---|---|
| passed, records | 85/120 (70.8%) | 99/120 (82.5%) | 88/120 (73.3%) | 95/120 (79.2%) |
| passed in 2 of 3 repeats | 29/40 | 33/40 | 30/40 | 32/40 |
| latency p50 / p95 (s) | 4.7 / 9.2 | 5.1 / 28.7 | 3.9 / 13.5 | 4.6 / 25.4 |

The difference is not significant. Paired on the 40 questions, exact McNemar:
`r125e` p = 0.125 (4 questions only the agent passed, none the other way, the
smallest p that 4 discordant pairs can give), `r126e` p = 0.625. The runs need
more questions. The ablation that caps the agent's search at k=4, like the
workflow's (`RR_MAX_SEARCH_K=4`), is not run yet (about 1 million Groq tokens),
so part of the gap may be search depth, not autonomy. More in
[docs/technical_deep_dive.md](docs/technical_deep_dive.md#9-results).

## Stack

- **Core**: Python 3.12, LangGraph (both systems are graphs; state checkpointed in SQLite), Pydantic.
- **Model**: Groq `openai/gpt-oss-120b` for the published runs (`.env.example` defaults to `openai/gpt-oss-20b`), with a rotation over up to 5 Groq keys; a local Ollama model is supported.
- **Retrieval**: grounded-rag (hybrid search with BM25 and embeddings, answer prompt, verifier), reached through `sys.path`.
- **Interfaces**: FastAPI with uvicorn, React 19 (Vite), Streamlit, a CLI.
- **Delivery**: Docker, Docker Hub, Render (API and UI), Streamlit Community Cloud.
- **Quality**: pytest (offline), GitHub Actions CI, Dependabot; optional LangSmith traces.
- **Measurement**: scoring, Wilson intervals, exact McNemar test and cost per correct answer in `eval/`; reports and charts with matplotlib.
- **Eval runs**: a cloud sandbox in Docker for the long runs, one Groq key per worker.

## Run it

Python 3.12. grounded-rag must sit next to this folder (or set `GROUNDED_RAG_PATH`)
with its indexes built and its embedder reachable (by default a local Ollama with
`nomic-embed-text`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock -r requirements-dev.txt
cp .env.example .env               # GROQ_API_KEY .. _5; RR_GROQ_MODEL (openai/gpt-oss-120b for the published runs)

python cli.py ask --system workflow "What were Apple's total net sales in fiscal 2025?"
python cli.py ask --system agent --approve "Quand s'appliquent les obligations haut risque ? Save a report."

python -m uvicorn api.main:app --port 8001      # React UI backend
cd web && npm install && npm run dev            # http://localhost:5181
python -m streamlit run app.py                  # Streamlit alternative, port 8501

python -m pytest -q                             # 307 tests, offline
```

An eval run, its resume, its check and its report: [docs/running_an_eval.md](docs/running_an_eval.md).
Docker, Render, Docker Hub and Streamlit Cloud: [docs/deploy.md](docs/deploy.md).
The React UI: [docs/react_ui.md](docs/react_ui.md).

## Safety

Tool allowlist with Pydantic, a whitelisted calculator, human approval before the
agent writes a file, a fence around retrieved text, a budget of 8 calls and 12,000
tokens per run, a loop guard, cite-or-refuse with verification, and key rotation
against free-tier limits. Each is described in the
[deep dive](docs/technical_deep_dive.md).

## Limits

- Three repeats of 40 questions on one model, 3 to 10 questions per type. The 120 records of a system are 40 questions asked three times, not 120 independent draws; on the 40 questions the 95% interval is 57-84% (workflow) and 68-91% (agent) in `r125e`.
- Wrong refusals are mostly retrieval misses: one short chunk (Article 113) holds the AI Act's application dates and rarely ranks. Both systems share that retrieval.
- Runs differ in grounded-rag version (`r125e` 80ef892, `r126e` e9535b2), and neither is the pin, 3b2de9f. Between 80ef892 and the pin the verifier reads negations and thousands separators more strictly; between the previous pin (3b09461) and 3b2de9f the chunks of both corpora are identical. Nothing has been rerun on the pin, so the published figures are not a measure of the pinned code. New runs record their versions in each record. Earlier runs (`r120e` to `r124e`) are not comparable ([decisions](docs/decisions.md)).
- Latency mostly measures free-tier rate-limit waits. Scoring is string and number matching; the opt-in LLM judge (`eval/judge.py`) has not been run on the published records. The verifier is lexical.
