import { useEffect, useRef } from "react";

// Human-in-the-loop gate for save_report: the agent run is paused server-side
// (LangGraph interrupt) until one of these buttons resumes it.
export default function ApprovalModal({ pending, busy, onApprove, onReject }) {
  const rejectRef = useRef(null);
  const dialogRef = useRef(null);

  useEffect(() => {
    rejectRef.current?.focus();
  }, []);

  const trap = (e) => {
    if (e.key !== "Tab") return;
    const items = dialogRef.current.querySelectorAll("button, [tabindex='0']");
    const first = items[0];
    const lastItem = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastItem.focus(); }
    else if (!e.shiftKey && document.activeElement === lastItem) { e.preventDefault(); first.focus(); }
  };

  const args = pending.args || {};
  return (
    <div className="modal-backdrop">
      <div className="modal card" role="dialog" aria-modal="true" aria-labelledby="approval-title"
        aria-describedby="approval-desc" ref={dialogRef} onKeyDown={trap}>
        <h2 id="approval-title">Approve <code>{pending.tool}</code>?</h2>
        <p id="approval-desc" className="hint">
          The agent wants to write this report to <code>reports/</code>. Nothing is saved unless you approve.
        </p>
        {args.title && <h3 className="report-title">{args.title}</h3>}
        <pre className="report-preview" tabIndex={0} aria-label="Report preview">
          {args.markdown || JSON.stringify(args, null, 2)}
        </pre>
        <div className="modal-actions">
          <button ref={rejectRef} className="secondary" disabled={busy} onClick={onReject}>Reject</button>
          <button className="primary" disabled={busy} onClick={onApprove}>{busy ? "Resuming…" : "Approve"}</button>
        </div>
      </div>
    </div>
  );
}
