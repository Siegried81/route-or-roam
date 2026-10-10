"""FastAPI backend for the React UI (web/): ask, approve, questions, runs and results.

    uvicorn api.main:app --port 8001

Thin HTTP layer: running the systems lives in api/service.py, results shaping in
api/results.py, and both reuse rr/ and eval/ instead of re-implementing them.
Endpoints are plain `def`, so FastAPI runs each (slow, blocking) model run in its
worker thread pool instead of blocking the event loop. Port 8001 leaves 8000 to
dreamjob and 8002 to grounded-rag's own API. When web/dist exists (after `npm run build`), the built
UI is served from `/` so one process is enough in production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import secrets

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from api import results, service
from eval.run_compare import QUESTIONS_PATH, RUNS_DIR, load_questions
from rr import settings
from rr.tools import TOOLS

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(title="route-or-roam API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5181", "http://127.0.0.1:5181"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Approve-Token"])

# Hosts a request may come from and still count as "this machine". Starlette's
# TestClient reports "testclient", which is the suite running on this machine.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _approver(request: Request, token: Optional[str]) -> str:
    """Who is approving, or an HTTP error. See settings.APPROVE_TOKEN.

    With a token configured, the header has to match it (constant-time, so a
    guess cannot be timed). Without one, the API is a local tool and only this
    machine may approve: an API started with `--host 0.0.0.0` and no token would
    otherwise let anyone on the network write a report in your name.
    """
    if settings.APPROVE_TOKEN:
        if not token or not secrets.compare_digest(token, settings.APPROVE_TOKEN):
            raise HTTPException(401, "approval needs a valid X-Approve-Token header")
        return "token"
    host = request.client.host if request.client else ""
    if host not in _LOCAL_HOSTS:
        raise HTTPException(403, "approval from another machine needs RR_APPROVE_TOKEN to be set")
    return f"local:{host}"


class AskRequest(BaseModel):
    """One question for one or both systems; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    system: Literal["workflow", "agent", "both"]
    question: str = Field(min_length=1, max_length=2000)
    injected_passage: Optional[str] = Field(default=None, max_length=5000)
    qid: Optional[str] = Field(default=None, pattern=r"^[\w.-]{1,40}$")


class ApproveRequest(BaseModel):
    """The human decision for a paused agent run."""

    model_config = ConfigDict(extra="forbid")
    thread_id: str = Field(min_length=1, max_length=200)
    approve: StrictBool


@app.get("/api/health")
def health() -> dict:
    """Liveness probe; touches no model and no file."""
    return {"status": "ok"}


def _groq_keys_available() -> bool:
    """True if ChatLLM will find a Groq key: our own, or grounded-rag's key ring it falls back to."""
    if settings.GROQ_API_KEYS:
        return True
    try:
        from rr.grounded import gr_config  # noqa: PLC0415 - same fallback as rr.llm.ChatLLM
    except Exception:
        return False
    return bool(getattr(gr_config, "GROQ_API_KEYS", None))


@app.get("/api/config")
def config() -> dict:
    """Model, budgets and tool allowlist the UI displays; key presence is a boolean only."""
    groq = settings.LLM_PROVIDER == "groq"
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.GROQ_MODEL if groq else settings.OLLAMA_MODEL,
        "key_available": _groq_keys_available() if groq else True,
        "budget": {"max_steps": settings.MAX_STEPS, "max_tokens": settings.MAX_TOKENS},
        "tools": [{"name": n, "description": s.description, "needs_approval": s.needs_approval}
                  for n, s in TOOLS.items()],
        "corpora": settings.CORPUS_DESCRIPTIONS,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Run the question; an agent paused on save_report comes back as awaiting_approval.

    "both" runs the workflow then the agent, one after the other, so neither
    system's latency is inflated by the other competing for the same rate limit.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "question is blank")
    injected = req.injected_passage or None
    qid = req.qid or "adhoc"
    systems = ["workflow", "agent"] if req.system == "both" else [req.system]
    out = {}
    for system in systems:
        run = service.run_workflow if system == "workflow" else service.run_agent
        out[system] = run(question, injected_passage=injected, qid=qid, run_id="api")
    status = "awaiting_approval" if any(r["status"] == "awaiting_approval" for r in out.values()) else "done"
    return {"status": status, "question": question, "results": out}


@app.post("/api/approve")
def approve(
    req: ApproveRequest,
    request: Request,
    x_approve_token: Optional[str] = Header(default=None, alias="X-Approve-Token"),
) -> dict:
    """Resume a paused agent run with the decision and return its (final or next) state.

    The decision is the one write this API performs, so it is the one call that
    checks who makes it (`_approver`) and says so in the response.
    """
    decided_by = _approver(request, x_approve_token)
    try:
        result = service.resume_agent(req.thread_id, req.approve)
    except service.UnknownThread:
        raise HTTPException(404, "no paused run with this thread_id") from None
    except service.Busy:
        raise HTTPException(409, "this run is already being resumed") from None
    return {"status": result["status"], "decided_by": decided_by, "results": {"agent": result}}


@app.get("/api/questions")
def questions() -> list[dict]:
    """The labelled eval questions, trimmed to what the picker needs (no answers or key facts)."""
    return [{"id": q["id"], "type": q["type"], "question": q["question"],
             "expect_refusal": bool(q.get("expect_refusal")),
             "injected": bool(q.get("injected_passage")),
             "injected_passage": q.get("injected_passage")}
            for q in load_questions(QUESTIONS_PATH)]


@app.get("/api/runs")
def runs() -> list[dict]:
    """runs/*.jsonl files, newest first."""
    return results.list_runs(RUNS_DIR) if RUNS_DIR.exists() else []


@app.get("/api/results")
def get_results(run: str = Query(..., min_length=1, max_length=200)) -> dict:
    """Scored comparison of one runs file (tolerates a partially written last line)."""
    path = results.safe_run_path(RUNS_DIR, run)
    if path is None:
        raise HTTPException(400, "run must be a plain *.jsonl file name")
    if not path.is_file():
        raise HTTPException(404, f"run {run} not found")
    try:
        return results.results_payload(path, load_questions(QUESTIONS_PATH))
    except (KeyError, TypeError, AttributeError):  # valid JSON lines that are not run records
        raise HTTPException(422, f"run {run} is not a valid runs file") from None


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
