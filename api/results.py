"""Turn a runs/*.jsonl file into the JSON the Results tab draws.

Scoring and aggregation are delegated to `eval.report.aggregate` (which scores
every record with `eval.score.score_record`), so the UI shows exactly the
numbers docs/results.md would show: one measurement definition, one place.
This module only reshapes tuples into named JSON fields and writes the takeaway
sentence from the aggregate, so it never assumes which system wins.
"""

from __future__ import annotations

import re
from pathlib import Path

from eval.report import TYPE_ORDER, aggregate, load_runs, paired_sentence

RUN_NAME_RE = re.compile(r"^[\w.-]+\.jsonl$")
# Below this gap in success (percentage points), or within the larger
# run-to-run spread, the two systems are reported as indistinguishable.
MIN_GAP_PP = 2.0


def safe_run_path(runs_dir: Path, name: str) -> Path | None:
    """Resolve a run file name inside `runs_dir`; None if the name is not a plain *.jsonl basename."""
    if not RUN_NAME_RE.match(name or "") or name.startswith("."):
        return None
    path = (runs_dir / name).resolve()
    return path if path.parent == runs_dir.resolve() else None


def list_runs(runs_dir: Path) -> list[dict]:
    """Every runs/*.jsonl file with its size, modification time and complete-line count."""
    out = []
    for path in runs_dir.glob("*.jsonl"):
        try:  # a file removed between glob and stat is skipped, not a 500
            st = path.stat()
            records = len(load_runs(path))
        except FileNotFoundError:
            continue
        out.append({"name": path.name, "bytes": st.st_size, "modified": st.st_mtime,
                    "records": records})
    return sorted(out, key=lambda r: r["modified"], reverse=True)


def _ms(triple) -> dict:
    """(mean, spread, repeats) from eval.report as named fields (rates in 0..1)."""
    mean, spread, repeats = triple
    return {"mean": mean, "spread": spread, "repeats": repeats}


def _ratio(a, b) -> float | None:
    """a / b, or None when either side is missing or b is zero."""
    return a / b if a is not None and b else None


def takeaway(systems: dict, paired: dict | None = None) -> str:
    """One sentence comparing workflow and agent, derived only from the aggregate.

    When both systems answered the same (question, repeat) pairs, the verdict
    comes from the exact McNemar test on those pairs (eval/stats.py): a gap is
    only called a difference when p < 0.05. Without pairs it falls back to the
    gap-versus-spread rule. The largest per-type gap is reported with its n,
    because one question can be worth 33 to 50 points in a type of 2 or 3.
    """
    if not systems:
        return "No scored records yet."
    if not {"workflow", "agent"} <= systems.keys():
        only = next(iter(systems))
        return f"Only the {only} has scored records so far (n={systems[only]['n']}); no comparison yet."
    w, a = systems["workflow"], systems["agent"]
    wm, am = w["overall"]["mean"], a["overall"]["mean"]
    if wm is None or am is None:
        return "Not enough scored records to compare the two systems."
    gap = 100 * (am - wm)
    calls = _ratio(a["llm_calls"], w["llm_calls"])
    lat = _ratio(a["p50_latency"], w["p50_latency"])
    cost = []
    if calls is not None:
        cost.append(f"{calls:.1f}x the LLM calls")
    if lat is not None:
        cost.append(f"{lat:.1f}x the median latency")
    cost_txt = f" The agent uses {' and '.join(cost)} of the workflow." if cost else ""
    if paired and paired["pairs"]:
        p = paired["p_value"]
        significant = p < 0.05
        test = f"exact McNemar p = {p:.2f} on {paired['pairs']} paired answers"
    else:
        band = max(MIN_GAP_PP, 100 * max(w["overall"]["spread"] or 0, a["overall"]["spread"] or 0))
        significant = abs(gap) >= band
        test = f"a gap under {band:.1f} pp"
    if not significant:
        head = (f"Workflow and agent are not significantly different ({100 * wm:.1f}% vs "
                f"{100 * am:.1f}% success; {test}).{cost_txt}")
    else:
        leader, trailer = ("agent", "workflow") if gap > 0 else ("workflow", "agent")
        hi, lo = (am, wm) if gap > 0 else (wm, am)
        head = (f"The {leader} succeeds more often than the {trailer} ({100 * hi:.1f}% vs "
                f"{100 * lo:.1f}%, +{abs(gap):.1f} pp; {test}).{cost_txt}")
    gaps = [(t, 100 * (a["by_type"][t]["mean"] - w["by_type"][t]["mean"]))
            for t in TYPE_ORDER
            if t in a["by_type"] and t in w["by_type"]
            and a["by_type"][t]["mean"] is not None and w["by_type"][t]["mean"] is not None]
    if gaps:
        t, g = max(gaps, key=lambda x: abs(x[1]))
        if abs(g) >= MIN_GAP_PP:
            n = min(w.get("n_by_type", {}).get(t, 0), a.get("n_by_type", {}).get(t, 0))
            size = f", n={n} each" if n else ""
            head += f" Largest gap by type: {t} ({'agent' if g > 0 else 'workflow'} +{abs(g):.0f} pp{size})."
    return head


def results_payload(runs_path: Path, questions: list[dict]) -> dict:
    """Aggregate one runs file into the Results tab JSON.

    `eval.report.load_runs` skips a line that does not parse, so a file still
    being appended to by a live evaluation is read up to its last complete record.
    """
    agg = aggregate(load_runs(runs_path), questions)
    systems = {}
    for name, m in agg["systems"].items():
        systems[name] = {
            "n": m["n"],
            "passed": m["passed"],
            "wilson": list(m["wilson"]) if m["wilson"] else None,
            "overall": _ms(m["overall"]),
            "by_type": {t: _ms(v) for t, v in m["by_type"].items()},
            "n_by_type": m["n_by_type"],
            "llm_calls": m["llm_calls"],
            "tokens_in": m["tokens_in"],
            "tokens_out": m["tokens_out"],
            "tokens": (m["tokens_in"] or 0) + (m["tokens_out"] or 0) if m["tokens_in"] is not None else None,
            "p50_latency": m["p50_latency"],
            "p95_latency": m["p95_latency"],
            "budget_pct": m["budget_pct"],
            "injection_pct": m["injection_pct"],
            "source_recall": m["source_recall"],
            "tool_use_pct": m["tool_use_pct"],
            "top_tags": [{"tag": t, "count": c} for t, c in m["top_tags"]],
        }
    types = [t for t in TYPE_ORDER if any(t in s["by_type"] for s in systems.values())]
    return {"run": runs_path.name, "records": sum(s["n"] for s in systems.values()),
            "unknown_qids": agg["unknown_qids"], "type_order": types, "systems": systems,
            "paired": agg["paired"], "paired_sentence": paired_sentence(agg["paired"]),
            "takeaway": takeaway(systems, agg["paired"])}
