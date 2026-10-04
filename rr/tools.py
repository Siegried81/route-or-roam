"""Allowlisted tool registry shared by both systems, with Pydantic-validated arguments.

Every tool call, whether chosen by the workflow's code or by the agent's model,
goes through `run_tool`: unknown names are rejected and counted as
hallucinated_tool, arguments are validated before anything runs, and
save_report refuses to write unless a human approved that exact call. Keeping
one entry point is what makes the two systems' tool traces comparable.

Search results get run-wide ids (S1, S2, ...) in order of first retrieval, so a
citation [S3] means the same passage in the answer, the trace and verification.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rr import fence, grounded, settings

INJECTED_SOURCE = "99_injected.txt"
Corpus = Literal["filings_sections", "ai_act_sections"]


class _Args(BaseModel):
    """Base for tool arguments: unknown keys are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


#: Most passages one search may return; larger requests are capped, not rejected.
MAX_SEARCH_K = 8


class SearchArgs(_Args):
    query: str = Field(min_length=1, max_length=500, description="What to look for.")
    corpus: Corpus = Field(description="Which document collection to search.")
    # No `le` bound in the schema on purpose: Groq validates tool calls against
    # the published JSON schema and rejects the whole call when gpt-oss asks for
    # k=10 against a maximum of 8. That rejection caused 7 of 17 agent failures
    # in the first measured run. The cap is enforced in code instead
    # (`capped_k`), so an over-eager k still searches, with at most 8 passages.
    k: int = Field(default=settings.SEARCH_K, ge=1,
                   description=f"Passages to return (at most {MAX_SEARCH_K}).")

    @property
    def capped_k(self) -> int:
        """`k` limited to MAX_SEARCH_K, the most passages one search returns."""
        return min(self.k, MAX_SEARCH_K)


class ListSectionsArgs(_Args):
    corpus: Corpus = Field(description="Which document collection to list.")


class CalculateArgs(_Args):
    expression: str = Field(min_length=1, max_length=200,
                            description="Arithmetic only, e.g. (416.2 - 391.0) / 391.0 * 100. "
                                        "No thousands separators, no units.")


class SaveReportArgs(_Args):
    title: str = Field(min_length=1, max_length=120)
    markdown: str = Field(min_length=1, max_length=20000)


@dataclass(frozen=True)
class ToolSpec:
    """A tool's argument model, the description the model sees, and whether a human must approve it."""

    args: type[_Args]
    description: str
    needs_approval: bool = False


TOOLS: dict[str, ToolSpec] = {
    "search_documents": ToolSpec(SearchArgs, "Search one corpus; returns passages with ids "
                                 "(S1, S2, ...) to cite. An empty result means nothing relevant was found."),
    "list_sections": ToolSpec(ListSectionsArgs, "List the section files of a corpus."),
    "calculate": ToolSpec(CalculateArgs, "Evaluate an arithmetic expression exactly."),
    "save_report": ToolSpec(SaveReportArgs, "Save a markdown report to disk. Only when the user "
                            "asks for a report; a human must approve it.", needs_approval=True),
}


def tool_schemas() -> list[dict]:
    """OpenAI-style function definitions for every allowlisted tool."""
    out = []
    for name, spec in TOOLS.items():
        params = spec.args.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        out.append({"type": "function",
                    "function": {"name": name, "description": spec.description, "parameters": params}})
    return out


# --- calculate: a whitelist AST evaluator (never eval) -------------------------
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max}


def safe_calculate(expression: str) -> float:
    """Evaluate +, -, *, /, //, %, ** and abs/round/min/max over numbers; reject anything else.

    Exponents are capped at 100 so a request like 9**9**9 cannot hang the run.
    """
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not an arithmetic expression: {exc.msg}") from exc
    return ev(tree)


# --- execution -----------------------------------------------------------------
@dataclass
class ToolResult:
    """Outcome of one tool call: `output` is what the model reads back."""

    ok: bool
    output: str
    error: str | None = None
    new_passages: list[dict] | None = None


def _register(passages: list[dict], retrieved, injected: str | None) -> tuple[list[dict], list[dict]]:
    """Give each hit a run-wide sid (reusing it if already seen); return (this call's hits, new ones)."""
    by_chunk = {p["chunk_id"]: p for p in passages}
    hits, new = [], []
    items = [(r.chunk.id, r.chunk.source, r.chunk.text, round(r.score, 4)) for r in retrieved]
    if injected:
        items.append(("injected", INJECTED_SOURCE, injected, 0.0))
    for chunk_id, source, text, score in items:
        p = by_chunk.get(chunk_id)
        if p is None:
            p = {"sid": f"S{len(passages) + len(new) + 1}", "chunk_id": chunk_id,
                 "source": source, "text": text, "score": score}
            new.append(p)
            by_chunk[chunk_id] = p
        hits.append(p)
    return hits, new


def _slug(title: str) -> str:
    """Filesystem-safe file stem from a title."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"


def run_tool(name: str, raw_args: dict, passages: list[dict], *,
             injected_passage: str | None = None, approved: bool = False) -> ToolResult:
    """Validate and run one allowlisted tool call; never raises for bad input.

    `passages` is the run's passage registry (read only here); newly seen
    passages are returned in `new_passages` for the caller to append to state.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(False, f"Unknown tool {name!r}. Allowed tools: {', '.join(TOOLS)}.",
                          error="hallucinated_tool")
    try:
        args = spec.args.model_validate(raw_args)
    except ValidationError as exc:
        msg = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        return ToolResult(False, f"Invalid arguments: {msg}", error=f"invalid_args: {msg}")
    if spec.needs_approval and not approved:
        return ToolResult(False, "Not executed: a human did not approve this call.",
                          error="approval_denied")
    try:
        if name == "search_documents":
            retrieved = grounded.search_corpus(args.query, args.corpus, args.capped_k)
            hits, new = _register(passages, retrieved, injected_passage)
            if not hits:
                return ToolResult(True, "No passages found: nothing in this corpus is relevant "
                                        "enough to this query.", new_passages=[])
            return ToolResult(True, fence.fence_all(hits), new_passages=new)
        if name == "list_sections":
            return ToolResult(True, "\n".join(grounded.list_corpus_sections(args.corpus)) or "(none)")
        if name == "calculate":
            return ToolResult(True, f"{args.expression} = {safe_calculate(args.expression):g}")
        if name == "save_report":
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = settings.REPORTS_DIR / f"{_slug(args.title)}.md"
            path.write_text(f"# {args.title}\n\n{args.markdown}\n", encoding="utf-8")
            return ToolResult(True, f"Saved report to reports/{path.name}")
    except Exception as exc:  # a failing tool is reported to the model, not raised
        return ToolResult(False, f"Tool error: {exc}", error=f"tool_error: {exc}")
    raise AssertionError(f"tool {name} registered but not implemented")
