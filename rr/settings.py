"""All tunables and paths for route-or-roam, read from the environment once.

Keeping the budget limits and provider settings in one module means the two
systems cannot silently run under different limits: both import them from here.
The .env file is loaded but never printed; only the variable names appear in
code and in .env.example.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# grounded-rag is reused in place (imported, never copied); see rr/grounded.py.
GROUNDED_RAG_PATH = Path(os.getenv("GROUNDED_RAG_PATH", str(ROOT.parent / "grounded-rag"))).resolve()

REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
CHECKPOINT_DB = RUNS_DIR / "checkpoints.sqlite"

# The two corpora both systems may search; anything else is rejected by validation.
CORPORA = ("filings_sections", "ai_act_sections")
CORPUS_DESCRIPTIONS = {
    "filings_sections": "Apple FY2025 Form 10-K, split into sections (English).",
    "ai_act_sections": "EU AI Act primer, split into sections (French).",
}

# --- Tracing (LangSmith, optional) -------------------------------------------
# LangGraph traces every node by itself once both of these are set; there is no
# client to build here. Read through settings like everything else so the two
# systems cannot end up traced under different projects.
# Note what it costs: traces leave the machine. See rr/tracing.py.
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "route-or-roam")

# --- LLM ---------------------------------------------------------------------
LLM_PROVIDER = os.getenv("RR_LLM_PROVIDER", "groq")  # groq | ollama
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("RR_GROQ_MODEL", "openai/gpt-oss-20b")
# Up to five keys; the client moves to the next one when a key is rate-limited.
GROQ_API_KEYS = [
    k
    for k in (os.getenv("GROQ_API_KEY", ""), *(os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 6)))
    if k
]
OLLAMA_BASE_URL = os.getenv("RR_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("RR_OLLAMA_MODEL", "llama3.2:3b")
LLM_TIMEOUT_S = 60
# Successful replies are cached on disk by an exact hash of the request
# (provider, model, messages, tools, options). Re-running an eval after an
# interruption then replays answered calls instead of spending free-tier quota.
# Turn it off (RR_LLM_CACHE=0) to measure run-to-run variance across repeats.
LLM_CACHE = os.getenv("RR_LLM_CACHE", "1") == "1"
LLM_CACHE_DIR = RUNS_DIR / "llm_cache"

# --- Budget (identical for both systems) ---------------------------------------
MAX_STEPS = int(os.getenv("RR_MAX_STEPS", "8"))       # LLM calls per run
MAX_TOKENS = int(os.getenv("RR_MAX_TOKENS", "12000"))  # prompt + completion tokens per run
SEARCH_K = 4
