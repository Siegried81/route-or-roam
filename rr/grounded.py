"""The single bridge to grounded-rag: path setup, corpus loading and search.

grounded-rag is imported in place from GROUNDED_RAG_PATH rather than copied, so
both systems retrieve, prompt and verify exactly as that project does, and any
fix made there applies here. This is the only module that touches sys.path; all
other rr modules import grounded-rag names from here.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from rr import settings

if not (settings.GROUNDED_RAG_PATH / "rag" / "retrieve.py").is_file():
    raise ImportError(
        f"grounded-rag not found at {settings.GROUNDED_RAG_PATH}; set GROUNDED_RAG_PATH."
    )
if str(settings.GROUNDED_RAG_PATH) not in sys.path:
    sys.path.insert(0, str(settings.GROUNDED_RAG_PATH))

import config as gr_config  # noqa: E402  (grounded-rag's config module)
from rag import answer as gr_answer  # noqa: E402
from rag.types import Chunk, Retrieved  # noqa: E402
from rag.verify import VerificationReport, verify_answer  # noqa: E402

# Re-exported so the rest of rr never copies grounded-rag's prompt text.
SYSTEM_PROMPT = gr_answer.SYSTEM_PROMPT
REFUSAL_MESSAGE = gr_answer.REFUSAL_MESSAGE
VERIFY_MIN_GROUNDING = gr_config.VERIFY_MIN_GROUNDING

__all__ = [
    "Chunk", "Retrieved", "VerificationReport", "SYSTEM_PROMPT", "REFUSAL_MESSAGE",
    "VERIFY_MIN_GROUNDING", "search_corpus", "list_corpus_sections", "verify",
]


@lru_cache(maxsize=None)
def _load(corpus: str):
    """Load (store, bm25, embedder) for a corpus once per process.

    Mirrors grounded-rag's cli/run_eval: BM25 is optional (an older index is
    dense-only) and the embedder comes from grounded-rag's own config, so the
    query is embedded in the same space the index was built in.
    """
    from rag.embed import get_embedder
    from rag.lexical import BM25Index
    from rag.store import VectorStore

    store = VectorStore.load(gr_config.index_path(corpus))
    try:
        bm25 = BM25Index.load(gr_config.bm25_path(corpus))
    except FileNotFoundError:
        bm25 = None
    return store, bm25, get_embedder()


def search_corpus(query: str, corpus: str, k: int) -> list[Retrieved]:
    """Run grounded-rag's hybrid retrieval; [] means nothing cleared the relevance bar.

    Tests monkeypatch this function, so no index, embedder or network is needed offline.
    """
    from rag.retrieve import retrieve

    store, bm25, embedder = _load(corpus)
    return retrieve(query, store, embedder, top_k=k, bm25=bm25)


def list_corpus_sections(corpus: str) -> list[str]:
    """Return the section file names of a corpus, sorted, from grounded-rag's data folder."""
    folder = gr_config.corpus_dir(corpus)
    return sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []


def verify(answer_text: str, sources: list[Retrieved]) -> VerificationReport:
    """grounded-rag's offline citation + grounding check, at grounded-rag's threshold."""
    return verify_answer(answer_text, sources, min_grounding=VERIFY_MIN_GROUNDING)
