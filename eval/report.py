"""Aggregate a runs file into docs/results.md (and docs/success_by_type.png).

Usage (from the project root):
    python -m eval.report --runs runs/r1.jsonl [--questions eval/questions.jsonl]
                          [--out docs/results.md] [--png docs/success_by_type.png]

Why it is built this way:
- Records are scored here, at report time, with ``eval.score`` against the
  current questions.jsonl: the measurement definition lives in one place and
  a re-score is free.
- Success is first computed per repeat, then summarised as mean +/- sample
  standard deviation across repeats. That spread is the run-to-run variance of
  a system on the same questions, which is the thing a single number hides.
- Latency uses nearest-rank percentiles (p50/p95), no interpolation, so every
  reported value is a latency that actually happened.
- matplotlib is optional: without it the markdown is still written and the
  chart is skipped with a note.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.run_compare import PROJECT_ROOT, QUESTIONS_PATH, load_questions
from eval.score import score_record

TYPE_ORDER = ["single_hop", "multi_hop", "numeric", "cross_corpus", "unanswerable", "injection"]


def load_runs(path: Path) -> list[dict]:
    """Read a runs jsonl file, skipping a possibly truncated last line."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def mean_spread(per_repeat: dict[int, list[bool]]) -> tuple[float | None, float | None, int]:
    """Mean and sample std of the per-repeat success rate, plus the repeat count.

    The std is 0.0 with a single repeat (no spread observable), None if no data.
    """
    rates = [sum(v) / len(v) for v in per_repeat.values() if v]
    if not rates:
        return None, None, 0
    spread = statistics.stdev(rates) if len(rates) > 1 else 0.0
    return statistics.mean(rates), spread, len(rates)


def _avg(values: list[float]) -> float | None:
    """Mean of non-None values, None if there are none."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(runs: list[dict], questions: list[dict]) -> dict:
    """Score every record and compute the per-system metrics shown in the report.

    Records whose qid is no longer in questions.jsonl are counted in
    ``unknown_qids`` and left out, instead of being scored against nothing.
    """
    by_id = {q["id"]: q for q in questions}
    out: dict = {"systems": {}, "unknown_qids": 0}
    grouped: dict[str, list[tuple[dict, dict, dict]]] = defaultdict(list)
    for rec in runs:
        q = by_id.get(rec.get("qid"))
        if q is None:
            out["unknown_qids"] += 1
            continue
        grouped[rec["system"]].append((rec, q, score_record(q, rec)))

    for system, items in sorted(grouped.items()):
        overall: dict[int, list[bool]] = defaultdict(list)
        by_type: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for rec, q, s in items:
            overall[rec.get("repeat", 0)].append(s["passed"])
            by_type[q["type"]][rec.get("repeat", 0)].append(s["passed"])
        injected = [s["injection_followed"] for rec, q, s in items if q.get("injected_passage")]
        tags = Counter(s["failure_tag"] for _, _, s in items if s["failure_tag"])
        n = len(items)
        out["systems"][system] = {
            "n": n,
            "overall": mean_spread(overall),
            "by_type": {t: mean_spread(by_type[t]) for t in TYPE_ORDER if t in by_type},
            "llm_calls": _avg([r.get("llm_calls") for r, _, _ in items]),
            "tokens_in": _avg([r.get("tokens_in") for r, _, _ in items]),
            "tokens_out": _avg([r.get("tokens_out") for r, _, _ in items]),
            "p50_latency": percentile([r["latency_s"] for r, _, _ in items if r.get("latency_s") is not None], 50),
            "p95_latency": percentile([r["latency_s"] for r, _, _ in items if r.get("latency_s") is not None], 95),
            "budget_pct": 100 * sum(bool(r.get("budget_exhausted")) for r, _, _ in items) / n,
            "injection_pct": 100 * sum(injected) / len(injected) if injected else None,
            "source_recall": _avg([s["source_recall"] for _, _, s in items]),
            "tool_use_pct": 100 * sum(s["tool_use_ok"] for _, _, s in items) / n,
            "top_tags": tags.most_common(3),
        }
    return out


def _pct(ms: tuple) -> str:
    """Format a (mean, spread, repeats) triple as 'xx.x% +/- y.y'."""
    mean, spread, _ = ms
    return "n/a" if mean is None else f"{100 * mean:.1f}% ± {100 * spread:.1f}"


def _num(v, fmt: str = "{:.1f}") -> str:
    """Format a number or 'n/a'."""
    return "n/a" if v is None else fmt.format(v)


def render_markdown(agg: dict, runs_name: str, png_note: str) -> str:
    """Build the results.md text: one table per system plus a by-type comparison."""
    lines = [f"# Results: `{runs_name}`", "",
             "Success = all key facts present (numbers ±1%), cited when answering, refusal exactly "
             "when expected, no forbidden string. Full definition: `eval/score.py`. "
             "Rates are mean ± sample std across repeats.", ""]
    if agg["unknown_qids"]:
        lines += [f"> {agg['unknown_qids']} record(s) skipped: qid not in questions.jsonl.", ""]
    for system, m in agg["systems"].items():
        tags = ", ".join(f"{t} ({c})" for t, c in m["top_tags"]) or "none"
        lines += [f"## {system}", "", "| metric | value |", "|---|---|",
                  f"| records | {m['n']} ({m['overall'][2]} repeats) |",
                  f"| overall success | {_pct(m['overall'])} |"]
        lines += [f"| success: {t} | {_pct(ms)} |" for t, ms in m["by_type"].items()]
        lines += [f"| avg LLM calls | {_num(m['llm_calls'], '{:.2f}')} |",
                  f"| avg tokens in / out | {_num(m['tokens_in'], '{:.0f}')} / {_num(m['tokens_out'], '{:.0f}')} |",
                  f"| latency p50 / p95 (s) | {_num(m['p50_latency'], '{:.2f}')} / {_num(m['p95_latency'], '{:.2f}')} |",
                  f"| budget exhausted | {m['budget_pct']:.1f}% |",
                  f"| injection followed | {_num(m['injection_pct'])}{'%' if m['injection_pct'] is not None else ''} |",
                  f"| source recall | {_num(m['source_recall'], '{:.2f}')} |",
                  f"| tool-use accuracy | {m['tool_use_pct']:.1f}% |",
                  f"| top failure tags | {tags} |", ""]
    systems = list(agg["systems"])
    if systems:
        lines += ["## Success by question type", "",
                  "| type | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        for t in TYPE_ORDER:
            cells = [_pct(agg["systems"][s]["by_type"][t]) if t in agg["systems"][s]["by_type"] else "n/a"
                     for s in systems]
            if any(c != "n/a" for c in cells):
                lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [png_note, ""]
    return "\n".join(lines)


def plot_success_by_type(agg: dict, png_path: Path) -> bool:
    """Grouped bars of success by type with std error bars; False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    systems = list(agg["systems"])
    types = [t for t in TYPE_ORDER if any(t in agg["systems"][s]["by_type"] for s in systems)]
    width = 0.8 / max(1, len(systems))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, s in enumerate(systems):
        stats = [agg["systems"][s]["by_type"].get(t, (None, None, 0)) for t in types]
        means = [100 * (m or 0) for m, _, _ in stats]
        errs = [100 * (sd or 0) for _, sd, _ in stats]
        xs = [x + (i - (len(systems) - 1) / 2) * width for x in range(len(types))]
        ax.bar(xs, means, width, yerr=errs, capsize=3, label=s)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Success by question type (mean ± std across repeats)")
    ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def build_report(runs_path: Path, questions_path: Path, out_md: Path, out_png: Path) -> dict:
    """Score, aggregate, write the markdown (and the chart if possible); return the aggregate."""
    agg = aggregate(load_runs(runs_path), load_questions(questions_path))
    if agg["systems"] and plot_success_by_type(agg, out_png):
        note = f"![success by type]({out_png.name})"
    else:
        note = "_Chart skipped: matplotlib not installed or no data._"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(agg, Path(runs_path).name, note), encoding="utf-8")
    return agg


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; see the module docstring for the flags."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "results.md")
    p.add_argument("--png", type=Path, default=PROJECT_ROOT / "docs" / "success_by_type.png")
    args = p.parse_args(argv)
    build_report(args.runs, args.questions, args.out, args.png)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
