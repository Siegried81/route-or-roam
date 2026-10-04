# React UI

A second front end next to the Streamlit app (`app.py`, unchanged). It has two parts:

- `api/`: a FastAPI backend on port **8001** (dreamjob uses 8000, grounded-rag 8002).
- `web/`: a Vite + React 18 app with two tabs, **Ask** and **Results**.

## Development (two terminals)

```bash
# 1. Backend, from the project root (reloads on Python changes)
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn api.main:app --port 8001 --reload

# 2. Frontend
cd web
npm install
npm run dev          # http://localhost:5181, /api is proxied to http://127.0.0.1:8001
```

Start uvicorn from the project root. `rr.grounded` puts grounded-rag at the front
of `sys.path`, and grounded-rag also has an `api` package; `uvicorn api.main:app`
imports this project's `api` before that happens, so the right package is used.

## Production (one process)

```bash
cd web && npm install && npm run build && cd ..
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
# open http://127.0.0.1:8001 : FastAPI serves web/dist when it exists
```

## Endpoints

| Method | Path | What it returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/config` | provider, model, budgets, tool allowlist, `key_available` (a boolean, never the key) |
| POST | `/api/ask` | `{system: workflow\|agent\|both, question, injected_passage?, qid?}` returns, per system, the `answer_question` record, the step trace and the retrieved passages. An agent that asks for `save_report` comes back as `status: "awaiting_approval"` with a `thread_id` and the pending report |
| POST | `/api/approve` | `{thread_id, approve: bool}` resumes the paused agent from the checkpointer and returns its final record (404 if nothing is waiting) |
| GET | `/api/questions` | the eval questions for the picker (no key facts) |
| GET | `/api/runs` | `runs/*.jsonl`, newest first |
| GET | `/api/results?run=<file>` | the comparison computed by `eval.report.aggregate` (scored with `eval.score`), plus a takeaway sentence derived from the numbers |

`/api/results` reads a file that a live evaluation is still writing: a last line
that is only half written is skipped.

## Notes on what is measured

- Records come from `rr.common.to_result`, the same contract the eval harness
  scores, so the Ask tab and the Results tab use the same fields.
- `both` runs the workflow, then the agent, one after the other, so neither
  system's latency includes waiting on the other's rate limit.
- Agent latency counts graph time only, not the time spent on the approval modal.
- The workflow trace comes from streaming the graph node by node (tokens per
  node are exact). The agent trace is rebuilt from its checkpointed messages,
  so only run-level token totals are known for the agent.
