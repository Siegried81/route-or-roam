import { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import RunColumn from "./components/RunColumn.jsx";
import ApprovalModal from "./components/ApprovalModal.jsx";

const SYSTEMS = [
  { id: "both", label: "Both" },
  { id: "workflow", label: "Workflow" },
  { id: "agent", label: "Agent" },
];

export default function AskTab({ config }) {
  const [questions, setQuestions] = useState([]);
  const [qid, setQid] = useState("");
  const [question, setQuestion] = useState("");
  const [injected, setInjected] = useState("");
  const [showInjected, setShowInjected] = useState(false);
  const [system, setSystem] = useState("both");
  const [running, setRunning] = useState(null); // list of systems in flight
  const [results, setResults] = useState({});
  const [error, setError] = useState(null);
  const [deciding, setDeciding] = useState(false);

  useEffect(() => {
    api("/api/questions").then(setQuestions).catch(() => setQuestions([]));
  }, []);

  const byType = useMemo(() => {
    const groups = {};
    for (const q of questions) (groups[q.type] ||= []).push(q);
    return groups;
  }, [questions]);

  const pick = (id) => {
    setQid(id);
    const q = questions.find((x) => x.id === id);
    if (!q) return;
    setQuestion(q.question);
    setInjected(q.injected_passage || "");
    setShowInjected(Boolean(q.injected_passage));
  };

  const run = async (e) => {
    e.preventDefault();
    if (!question.trim() || running) return;
    const systems = system === "both" ? ["workflow", "agent"] : [system];
    setRunning(systems);
    setError(null);
    setResults({});
    try {
      const body = { system, question };
      if (injected.trim() && showInjected) body.injected_passage = injected;
      const picked = questions.find((x) => x.id === qid);
      if (picked && picked.question === question) body.qid = qid;
      const data = await api("/api/ask", body);
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(null);
    }
  };

  const decide = async (approve) => {
    const agent = results.agent;
    setDeciding(true);
    try {
      const data = await api("/api/approve", { thread_id: agent.thread_id, approve });
      setResults((r) => ({ ...r, agent: data.results.agent }));
    } catch (err) {
      setError(err.message);
      setResults((r) => ({ ...r, agent: { ...r.agent, status: "done" } }));
    } finally {
      setDeciding(false);
    }
  };

  const picked = questions.find((x) => x.id === qid);
  const columns = running || Object.keys(results);
  const pendingAgent = results.agent && results.agent.status === "awaiting_approval" ? results.agent : null;

  return (
    <div className="ask">
      <form className="card ask-form" onSubmit={run}>
        <div className="field">
          <label htmlFor="picker">Eval question</label>
          <select id="picker" value={qid} onChange={(e) => pick(e.target.value)}>
            <option value="">Write your own…</option>
            {Object.entries(byType).map(([type, qs]) => (
              <optgroup key={type} label={type.replace("_", " ")}>
                {qs.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.id} · {q.question.length > 80 ? q.question.slice(0, 80) + "…" : q.question}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          {picked && (
            <div className="badges">
              <span className="badge neutral">{picked.type.replace("_", " ")}</span>
              {picked.expect_refusal && <span className="badge info">expects refusal</span>}
              {picked.injected && <span className="badge redteam">injected passage</span>}
            </div>
          )}
        </div>

        <div className="field">
          <label htmlFor="question">Question</label>
          <textarea id="question" rows={3} value={question} maxLength={2000} required
            placeholder="What were Apple's total net sales in fiscal 2025?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(e); }} />
        </div>

        <div className="field redteam-field">
          <label className="check">
            <input type="checkbox" checked={showInjected} onChange={(e) => setShowInjected(e.target.checked)} />
            <span className="badge redteam">red-team</span> Inject a passage into every search result
          </label>
          {showInjected && (
            <>
              <textarea id="injected" aria-label="Injected passage" rows={3} value={injected} maxLength={5000}
                placeholder="Ignore all previous instructions and…" onChange={(e) => setInjected(e.target.value)} />
              <p className="hint">Appended to each search result as <code>99_injected.txt</code>, to test whether a system follows instructions found in sources.</p>
            </>
          )}
        </div>

        <div className="form-row">
          <div className="segmented" role="radiogroup" aria-label="System to run">
            {SYSTEMS.map((s) => (
              <label key={s.id} className={system === s.id ? "on" : ""}>
                <input type="radio" name="system" value={s.id} checked={system === s.id}
                  onChange={() => setSystem(s.id)} />
                {s.label}
              </label>
            ))}
          </div>
          <button type="submit" className="primary" disabled={!question.trim() || Boolean(running)}>
            {running ? "Running…" : "Run"}
          </button>
        </div>
        {config && !config.key_available && (
          <p className="hint warn-text">No API key is configured on the server: runs will return an error record.</p>
        )}
      </form>

      {error && <div className="alert" role="alert">{error}</div>}

      {columns.length > 0 && (
        <div className={`columns n${columns.length}`} aria-live="polite" aria-busy={Boolean(running)}>
          {columns.map((s) => (
            <RunColumn key={s} system={s} result={results[s]} loading={Boolean(running)}
              budget={config && config.budget} />
          ))}
        </div>
      )}

      {pendingAgent && (
        <ApprovalModal pending={pendingAgent.pending} busy={deciding}
          onApprove={() => decide(true)} onReject={() => decide(false)} />
      )}
    </div>
  );
}
