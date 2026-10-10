import { useEffect, useRef, useState } from "react";
import { approveToken } from "../api";

// Human-in-the-loop gate for save_report: the agent run is paused server-side
// (LangGraph interrupt) until one of these buttons resumes it.
// Keyboard: focus starts on Reject (the safe default), Tab cycles inside the
// dialog, Escape rejects, and focus returns where it was when the dialog closes.
// While a decision is being sent both buttons are disabled, so the dialog itself
// holds focus instead of letting it fall back to the page behind.
export default function ApprovalModal({ pending, busy, error, onApprove, onReject }) {
  const rejectRef = useRef(null);
  const dialogRef = useRef(null);
  // Shown only once the server has asked for it (a 401 on approve): a local
  // run never needs a token, and a field nobody needs is a field people fill.
  const [token, setToken] = useState(approveToken.get());
  const needsToken = Boolean(error && /token/i.test(error)) || Boolean(token);

  useEffect(() => {
    const previous = document.activeElement;
    rejectRef.current?.focus();
    return () => previous?.focus?.();
  }, []);

  useEffect(() => {
    if (busy) dialogRef.current?.focus();
    else rejectRef.current?.focus();
  }, [busy]);

  const onKeyDown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (!busy) onReject();
      return;
    }
    if (e.key !== "Tab") return;
    const items = dialogRef.current.querySelectorAll("button:not(:disabled), [tabindex='0']");
    if (!items.length) { e.preventDefault(); return; }
    const first = items[0];
    const lastItem = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastItem.focus(); }
    else if (!e.shiftKey && document.activeElement === lastItem) { e.preventDefault(); first.focus(); }
  };

  const args = pending.args || {};
  return (
    <div className="modal-backdrop">
      <div className="modal card" role="dialog" aria-modal="true" aria-labelledby="approval-title"
        aria-describedby="approval-desc" ref={dialogRef} tabIndex={-1} onKeyDown={onKeyDown}
        aria-busy={busy}>
        <h2 id="approval-title">Approve <code>{pending.tool}</code>?</h2>
        <p id="approval-desc" className="hint">
          The agent wants to write this report to <code>reports/</code>. Nothing is saved unless you approve.
          Escape rejects.
        </p>
        {args.title && <h3 className="report-title">{args.title}</h3>}
        <pre className="report-preview" tabIndex={0} aria-label="Report preview">
          {args.markdown || JSON.stringify(args, null, 2)}
        </pre>
        {error && <div className="alert" role="alert">{error}</div>}
        {needsToken && (
          <div className="field">
            <label htmlFor="approve-token">Approval token (this server requires one)</label>
            <input id="approve-token" type="password" value={token} autoComplete="off"
              onChange={(e) => { setToken(e.target.value); approveToken.set(e.target.value); }} />
          </div>
        )}
        <div className="modal-actions">
          <button ref={rejectRef} className="secondary" disabled={busy} onClick={onReject}>Reject</button>
          <button className="primary" disabled={busy} onClick={onApprove}>{busy ? "Resuming…" : "Approve"}</button>
        </div>
      </div>
    </div>
  );
}
