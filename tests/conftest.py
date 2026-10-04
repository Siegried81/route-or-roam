"""Shared offline fixtures: no network, canned retrieval, temp report dir, in-memory checkpoints.

grounded-rag's retrieval needs a built index and an embedding server, so every
test replaces `rr.grounded.search_corpus` with canned passages; the real
verifier, prompts and Chunk/Retrieved types from grounded-rag are still used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rr import grounded, settings  # noqa: E402

CORPUS = {
    "filings_sections": [
        ("05_mdna.txt", "Apple total net sales were 416 billion dollars in fiscal 2025, "
                        "compared with 391 billion dollars in fiscal 2024."),
        ("02_risk_factors.txt", "Apple faces intense competition and supply chain risks "
                                "in many markets."),
    ],
    "ai_act_sections": [
        ("02_timeline.txt", "Les obligations pour les systemes a haut risque s'appliquent "
                            "a partir d'aout 2026."),
    ],
}


def fake_hits(corpus: str, k: int = 4) -> list:
    """Canned grounded-rag Retrieved results for a corpus."""
    return [grounded.Retrieved(grounded.Chunk(id=f"{corpus}-{i}", text=text, source=src,
                                              ordinal=i, corpus=corpus), 0.8 - i * 0.1)
            for i, (src, text) in enumerate(CORPUS.get(corpus, [])[:k])]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Block HTTP, fake retrieval, redirect reports and keep checkpoints in memory."""
    import requests
    from langgraph.checkpoint.memory import MemorySaver

    from rr import agent

    def no_network(*a, **k):
        raise AssertionError("network call attempted in a test")

    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(grounded, "search_corpus", lambda q, corpus, k: fake_hits(corpus, k))
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "LLM_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(settings, "LLM_CACHE", False)
    monkeypatch.setattr(agent, "_CHECKPOINTER", MemorySaver())
    monkeypatch.setattr("time.sleep", lambda s: None)
