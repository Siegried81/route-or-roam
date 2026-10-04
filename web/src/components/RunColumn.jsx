import { SYSTEM_LABEL, fmt } from "../api.js";
import Answer from "./Answer.jsx";
import Timeline from "./Timeline.jsx";

export default function RunColumn({ system, result, loading, budget }) {
  const cls = system === "workflow" ? "sys-a" : "sys-b";
  if (!result) {
    return (
      <section className={`card column ${cls}`} aria-label={SYSTEM_LABEL[system]}>
        <ColumnHeader system={system} status={loading ? "running" : "idle"} />
        {loading && <div className="skeleton" aria-hidden="true"><div /><div /><div /></div>}
        {loading && <p className="hint">Running {system}… live model calls can take 20 to 60 seconds.</p>}
      </section>
    );
  }
  const r = result.record;
  const tokens = r.tokens_in + r.tokens_out;
  return (
    <section className={`card column ${cls}`} aria-label={SYSTEM_LABEL[system]}>
      <ColumnHeader system={system} status={result.status} />

      <div className="chips">
        <Chip label="LLM calls" value={budget ? `${r.llm_calls} / ${budget.max_steps}` : r.llm_calls} />
        <Chip label="tokens" value={fmt.int(tokens)} title={`${r.tokens_in} in / ${r.tokens_out} out`} />
        <Chip label="latency" value={fmt.sec(r.latency_s)} />
        <Chip label="tool calls" value={r.tool_calls.length} />
      </div>

      <div className="badges">
        {r.refused && <span className="badge info">refused</span>}
        {r.budget_exhausted && <span className="badge warn">budget exhausted</span>}
        {r.injection_flags > 0 && (
          <span className="badge redteam">{r.injection_flags} injection flag{r.injection_flags > 1 ? "s" : ""}</span>
        )}
        {r.hallucinated_tools > 0 && <span className="badge warn">{r.hallucinated_tools} unknown tool call(s)</span>}
        {result.status === "done" && !r.error && !r.refused && r.citations.length === 0 && (
          <span className="badge warn">uncited</span>
        )}
        {result.status === "done" && r.citations.length > 0 && (
          <span className="badge ok">cited {r.citations.join(", ")}</span>
        )}
      </div>

      {r.error && <div className="alert" role="alert">{r.error}</div>}

      {result.status === "done" ? (
        <Answer text={r.answer} passages={result.passages} cited={r.citations} />
      ) : (
        <div className="answer pending">Waiting for a human decision on <code>{result.pending.tool}</code>…</div>
      )}

      <h3 className="section-title">Trace</h3>
      <Timeline steps={result.trace} system={system} passages={result.passages} />
    </section>
  );
}

function ColumnHeader({ system, status }) {
  const label = { done: "done", running: "running", awaiting_approval: "awaiting approval", idle: "" }[status];
  return (
    <header className="column-head">
      <h2><span className="swatch" aria-hidden="true" />{SYSTEM_LABEL[system]}</h2>
      {label && <span className={`status ${status}`}>{label}</span>}
    </header>
  );
}

function Chip({ label, value, title }) {
  return (
    <span className="chip" title={title}>
      <span className="chip-value">{value}</span>
      <span className="chip-label">{label}</span>
    </span>
  );
}
