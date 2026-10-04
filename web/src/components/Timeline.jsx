import { fmt } from "../api.js";

const KIND = {
  plan: { label: "Plan", icon: "◇" },
  search: { label: "Search", icon: "⌕" },
  calculate: { label: "Calculate", icon: "∑" },
  answer: { label: "Answer", icon: "✎" },
  verify: { label: "Verify", icon: "✓" },
  stop: { label: "Stopped", icon: "■" },
  thought: { label: "Thought", icon: "💭" },
  tool: { label: "Tool call", icon: "⚙" },
  feedback: { label: "Verifier feedback", icon: "↺" },
  approval: { label: "Human approval", icon: "✋" },
  trigger: { label: "Trigger", icon: "▶" },
};

export default function Timeline({ steps, system }) {
  if (!steps || steps.length === 0) return <p className="hint">No steps recorded.</p>;
  // Agent: an assistant turn and the tool calls it requested form one loop iteration.
  let turn = 0;
  return (
    <ol className="timeline">
      {steps.map((s, i) => {
        if (system === "agent" && (s.kind === "thought" || s.kind === "answer")) turn += 1;
        const meta = KIND[s.kind] || { label: s.kind, icon: "•" };
        const alert = s.events.length > 0;
        return (
          <li key={i} className={`step kind-${s.kind} ${alert ? "has-event" : ""}`}>
            <span className="step-dot" aria-hidden="true">{meta.icon}</span>
            <div className="step-body">
              <div className="step-head">
                <span className="step-title">
                  {meta.label}
                  {system === "agent" && s.node === "agent" && <span className="muted"> · turn {turn}</span>}
                </span>
                <span className="step-chips">
                  {s.llm_calls > 0 && <span className="mini">{s.llm_calls} LLM</span>}
                  {s.tokens > 0 && <span className="mini">{fmt.int(s.tokens)} tok</span>}
                </span>
              </div>
              <StepDetail step={s} />
              {s.tool_calls.map((c, j) => <ToolCard key={j} call={c} />)}
              {s.kind === "tool" && s.detail.observation && (
                <details className="observation">
                  <summary>Observation</summary>
                  <pre>{s.detail.observation}</pre>
                </details>
              )}
              {s.events.map((e, j) => (
                <div key={j} className={`event ev-${e.type}`} role="note">
                  <strong>{e.type.replace(/_/g, " ")}</strong> {e.message}
                </div>
              ))}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StepDetail({ step }) {
  const d = step.detail || {};
  switch (step.kind) {
    case "plan":
      if (!d.plan) return null;
      return (
        <div className="detail">
          {d.plan.fallback && <div className="event ev-fallback"><strong>fallback</strong> The plan was invalid; searching the literal question in every corpus.</div>}
          <div className="kv"><span>corpora</span>{d.plan.corpora.map((c) => <code key={c}>{c}</code>)}</div>
          <div className="kv"><span>queries</span>
            <ul className="plain">{d.plan.sub_queries.map((q, i) => <li key={i}>{q}</li>)}</ul>
          </div>
          <div className="kv"><span>needs math</span>{d.plan.needs_math ? "yes" : "no"}</div>
        </div>
      );
    case "search":
      return <p className="detail muted">{d.new_passages} new passage(s) registered</p>;
    case "calculate":
      return <p className="detail">{d.calc_output ? <code>{d.calc_output}</code> : <span className="muted">no valid expression</span>}</p>;
    case "answer":
    case "stop": {
      const text = d.answer || d.content;
      if (!text) return null;
      return (
        <details className="detail draft" open={step.kind === "stop"}>
          <summary>{step.kind === "answer" ? `Draft${d.attempt > 1 ? ` (attempt ${d.attempt})` : ""}` : "Message"}</summary>
          <p>{text}</p>
        </details>
      );
    }
    case "verify": {
      const v = d.verify || {};
      return (
        <div className="detail">
          <span className={`badge ${v.ok ? "ok" : "warn"}`}>{v.ok ? "verified" : "failed"}</span>
          {v.grounding !== undefined && <span className="mini">grounding {fmt.num(v.grounding, 2)}</span>}
          {d.refused && <span className="badge info">refusal</span>}
          {v.invalid_citations && v.invalid_citations.length > 0 && (
            <span className="badge warn">invalid citations {v.invalid_citations.join(", ")}</span>
          )}
          {v.uncited_sentences && v.uncited_sentences.length > 0 && (
            <span className="mini">{v.uncited_sentences.length} uncited sentence(s)</span>
          )}
        </div>
      );
    }
    case "thought":
      return (
        <div className="detail">
          {d.content ? <p className="thought">{d.content}</p> : <p className="muted">(no visible reasoning)</p>}
          {d.requested.length > 0 && (
            <div className="kv"><span>requests</span>{d.requested.map((n, i) => <code key={i}>{n}</code>)}</div>
          )}
        </div>
      );
    case "feedback":
      return <p className="detail thought">{d.content}</p>;
    case "approval":
      return <p className="detail">Waiting on <code>{d.pending.tool}</code> “{d.pending.args && d.pending.args.title}”.</p>;
    default:
      return d.content ? <p className="detail">{d.content}</p> : null;
  }
}

function ToolCard({ call }) {
  const ok = call.ok;
  return (
    <div className={`tool-card ${ok ? "ok" : "bad"}`}>
      <div className="tool-head">
        <code className="tool-name">{call.name}</code>
        <span className={`badge ${ok ? "ok" : "bad"}`}>{ok ? "ok" : "error"}</span>
      </div>
      <dl className="args">
        {Object.entries(call.args || {}).map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{typeof v === "string" ? v : JSON.stringify(v)}</dd></div>
        ))}
      </dl>
      {call.error && <div className="tool-error">{call.error}</div>}
    </div>
  );
}
