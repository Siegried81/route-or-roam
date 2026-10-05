# React UI

A second front end next to the Streamlit app (`app.py`, unchanged). It has two parts:

- `api/`: a FastAPI backend on port **8001**. The ports are fixed so the three
  local apps run side by side: dreamjob owns 8000 (and Vite's 5173),
  grounded-rag's API owns 8002 (UI on 5180), this one 8001 (UI on 5181).
- `web/`: a Vite + React 18 app with two tabs, **Ask** and **Results**.

## Development (two terminals)

```bash
# 1. Backend, from the project root (reloads on Python changes)
.venv/bin/python -m pip install -r requirements.txt            # Windows: .venv/Scripts/python
.venv/bin/python -m uvicorn api.main:app --port 8001 --reload

# 2. Frontend
cd web
npm install          # npm comes from nvm: interactive shell, `which npm` under ~/.nvm
npm run dev          # http://localhost:5181, /api is proxied to http://127.0.0.1:8001
```

Start uvicorn from the project root. `rr.grounded` puts grounded-rag at the front
of `sys.path`, and grounded-rag also has an `api` package; `uvicorn api.main:app`
imports this project's `api` before that happens, so the right package is used.

## Production (one process)

```bash
cd web && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
# open http://127.0.0.1:8001 : FastAPI serves web/dist when it exists
```

`web/dist` is not committed: until `npm run build` has run, `/` is a 404 and only
the `/api/*` routes answer.

## Endpoints

| Method | Path | What it returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/config` | provider, model, budgets, tool allowlist, `key_available` (a boolean, never the key; true when either this project's keys or grounded-rag's key ring is set) |
| POST | `/api/ask` | `{system: workflow\|agent\|both, question, injected_passage?, qid?}` returns, per system, the `answer_question` record, the step trace and the retrieved passages. An agent that asks for `save_report` comes back as `status: "awaiting_approval"` with a `thread_id` and the pending report. 422 on invalid input |
| POST | `/api/approve` | `{thread_id, approve: bool}` resumes the paused agent from the checkpointer and returns its next state: the final record, or `awaiting_approval` again when the agent asks for another report. 404 if nothing is waiting under that id, 409 if the same run is already being resumed |
| GET | `/api/questions` | the eval questions for the picker (no key facts) |
| GET | `/api/runs` | `runs/*.jsonl`, newest first |
| GET | `/api/results?run=<file>` | the comparison computed by `eval.report.aggregate` (scored with `eval.score`), plus a takeaway sentence derived from the numbers. 400 for a name that is not a plain `*.jsonl`, 404 if the file is missing, 422 if its lines are not run records |

`/api/results` reads a file that a live evaluation is still writing: a last line
that is only half written is skipped. Per system it returns `n`, `passed`,
`wilson` (the 95% interval as `[low, high]`, rates in 0..1), `overall` and
`by_type` as `{mean, spread, repeats}` (`spread` is `null` with one repeat),
`n_by_type`, calls, tokens, latency percentiles and the top failure tags; at the
top level `paired` (the McNemar cells and p-value from `eval/stats.py`) and
`paired_sentence`, the plain-language version the page shows. The takeaway calls
a difference only when the McNemar p-value is under 0.05.

After an API restart, `/api/approve` can still resume a run that was paused
before it: the state is in the SQLite checkpointer. That fallback only accepts
thread ids this API created (prefix `api:`), and a resume that fails keeps the
checkpointed state so the error record still shows the calls and tokens spent.

## Notes on what is measured

- Records come from `rr.common.to_result`, the same contract the eval harness
  scores, so the Ask tab and the Results tab use the same fields.
- `both` runs the workflow, then the agent, one after the other, so neither
  system's latency includes waiting on the other's rate limit. The page sends
  "Both" as two requests (workflow, then agent) so each column fills in as soon
  as its system finishes, with its own elapsed timer meanwhile; the server runs
  the two in the same order, so the timing semantics are the same either way.
- Agent latency counts graph time only, not the time spent on the approval modal.
- The workflow trace comes from streaming the graph node by node (tokens per
  node are exact). The agent trace is rebuilt from its checkpointed messages,
  so only run-level token totals are known for the agent.

## Page behaviour

- Both tabs stay mounted; switching tabs never drops a run in flight or an
  agent paused for approval.
- Approval modal: focus starts on Reject (the safe default) and Tab cycles
  inside the dialog; Escape rejects; focus returns where it was when the dialog
  closes. An error from `/api/approve` is shown inside the modal and the
  decision can be retried; only a 404 (the run is gone) closes it, since any
  other failure leaves the run paused on the server.
- Results tab: the Wilson interval, n per type and the paired sentence sit next
  to the rates; under 600px the chart gives way to the table.
- An AI disclosure footer is always visible: answers come from a model and can
  be wrong; check the cited sources.
