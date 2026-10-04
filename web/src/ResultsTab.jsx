import { useCallback, useEffect, useState } from "react";
import { api, fmt, SYSTEM_LABEL } from "./api.js";
import BarChart from "./components/BarChart.jsx";

// "better" marks which direction wins, so the better of the two cells can be bolded.
const KPIS = [
  { key: "overall", label: "Overall success", get: (s) => s.overall.mean,
    show: (s) => `${fmt.pct(s.overall.mean)} ± ${fmt.pct(s.overall.spread || 0)}`, better: "high" },
  { key: "llm_calls", label: "Avg LLM calls", get: (s) => s.llm_calls, show: (s) => fmt.num(s.llm_calls, 2), better: "low" },
  { key: "tokens", label: "Avg tokens (in + out)", get: (s) => s.tokens, show: (s) => fmt.int(s.tokens), better: "low" },
  { key: "p50", label: "Latency p50", get: (s) => s.p50_latency, show: (s) => fmt.sec(s.p50_latency), better: "low" },
  { key: "p95", label: "Latency p95", get: (s) => s.p95_latency, show: (s) => fmt.sec(s.p95_latency), better: "low" },
  { key: "budget", label: "Budget exhausted", get: (s) => s.budget_pct, show: (s) => fmt.pctRaw(s.budget_pct), better: "low" },
  { key: "inj", label: "Injection followed", get: (s) => s.injection_pct, show: (s) => fmt.pctRaw(s.injection_pct), better: "low" },
  { key: "recall", label: "Source recall", get: (s) => s.source_recall, show: (s) => fmt.num(s.source_recall, 2), better: "high" },
  { key: "tooluse", label: "Tool-use accuracy", get: (s) => s.tool_use_pct, show: (s) => fmt.pctRaw(s.tool_use_pct), better: "high" },
];
const ORDER = ["workflow", "agent"];

export default function ResultsTab() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [asTable, setAsTable] = useState(false);

  useEffect(() => {
    api("/api/runs").then((rs) => {
      setRuns(rs);
      if (rs.length) setRun((cur) => cur || rs[0].name);
    }).catch((e) => setError(e.message));
  }, []);

  const load = useCallback(() => {
    if (!run) return;
    setLoading(true);
    setError(null);
    api(`/api/results?run=${encodeURIComponent(run)}`)
      .then(setData)
      .catch((e) => { setError(e.message); setData(null); })
      .finally(() => setLoading(false));
  }, [run]);

  useEffect(() => { load(); }, [load]);

  const systems = data ? ORDER.filter((s) => data.systems[s]) : [];

  return (
    <div className="results">
      <div className="card toolbar">
        <div className="field inline">
          <label htmlFor="run">Run</label>
          <select id="run" value={run} onChange={(e) => setRun(e.target.value)} disabled={!runs.length}>
            {runs.length === 0 && <option>No runs/*.jsonl yet</option>}
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name} · {r.records} records</option>
            ))}
          </select>
        </div>
        <button className="secondary" onClick={load} disabled={!run || loading}>{loading ? "Loading…" : "Refresh"}</button>
        {data && (
          <span className="muted">
            {data.records} scored records{data.unknown_qids ? ` · ${data.unknown_qids} skipped (unknown qid)` : ""}
          </span>
        )}
      </div>

      {error && <div className="alert" role="alert">{error}</div>}

      {data && (
        <>
          <p className="takeaway card" aria-live="polite">{data.takeaway}</p>

          <section className="card">
            <div className="section-head">
              <h2>Success by question type</h2>
              <label className="check small">
                <input type="checkbox" checked={asTable} onChange={(e) => setAsTable(e.target.checked)} /> Show as table
              </label>
            </div>
            {data.type_order.length === 0 ? <p className="hint">No scored records yet.</p>
              : asTable ? <TypeTable data={data} systems={systems} /> : <BarChart data={data} />}
          </section>

          <div className="grid-2">
            <section className="card">
              <h2>Key metrics</h2>
              <div className="table-scroll">
                <table className="kpi">
                  <thead>
                    <tr>
                      <th scope="col">Metric</th>
                      {systems.map((s) => (
                        <th key={s} scope="col" className={s === "workflow" ? "sys-a" : "sys-b"}>
                          <span className="swatch" aria-hidden="true" />{s}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr><th scope="row">Records</th>{systems.map((s) => <td key={s}>{data.systems[s].n}</td>)}</tr>
                    {KPIS.map((k) => {
                      const vals = systems.map((s) => k.get(data.systems[s]));
                      const known = vals.filter((v) => v !== null && v !== undefined);
                      const best = known.length === 2 && known[0] !== known[1]
                        ? (k.better === "high" ? Math.max(...known) : Math.min(...known)) : null;
                      return (
                        <tr key={k.key}>
                          <th scope="row">{k.label}</th>
                          {systems.map((s, i) => (
                            <td key={s} className={vals[i] === best ? "best" : ""}>
                              {k.show(data.systems[s])}
                              {vals[i] === best && <span className="sr-only"> (better)</span>}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="hint">
                Bold = better of the two. Success: every key fact present (numbers ±1%), cited when answering,
                refusal exactly when expected, no forbidden string (eval/score.py).
              </p>
            </section>

            <section className="card">
              <h2>Top failure tags</h2>
              <div className="tags-grid">
                {systems.map((s) => (
                  <div key={s} className={s === "workflow" ? "sys-a" : "sys-b"}>
                    <h3><span className="swatch" aria-hidden="true" />{SYSTEM_LABEL[s]}</h3>
                    {data.systems[s].top_tags.length === 0 ? <p className="muted">No failures.</p> : (
                      <ul className="tag-list">
                        {data.systems[s].top_tags.map((t) => (
                          <li key={t.tag}>
                            <span className="tag-name">{t.tag.replace(/_/g, " ")}</span>
                            <span className="tag-bar" aria-hidden="true">
                              <span style={{ width: `${(100 * t.count) / data.systems[s].n}%` }} />
                            </span>
                            <span className="tag-count">{t.count} / {data.systems[s].n}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
              <p className="hint">
                Primary tag per failed record (priority: error, injection followed, loop or budget, false refusal,
                over-answer, bad tool args, wrong number, missed hop, uncited).
              </p>
            </section>
          </div>
        </>
      )}
    </div>
  );
}

function TypeTable({ data, systems }) {
  return (
    <div className="table-scroll">
      <table className="kpi">
        <thead>
          <tr><th scope="col">Type</th>{systems.map((s) => <th key={s} scope="col">{s}</th>)}</tr>
        </thead>
        <tbody>
          {data.type_order.map((t) => (
            <tr key={t}>
              <th scope="row">{t.replace("_", " ")}</th>
              {systems.map((s) => {
                const st = data.systems[s].by_type[t];
                return <td key={s}>{st ? `${fmt.pct(st.mean)} ± ${fmt.pct(st.spread || 0)}` : "n/a"}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
