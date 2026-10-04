// Final answer with [S#] citations turned into chips; the models sometimes write 【S#】 instead.
const CITE_RE = /\[(S\d+)\]|【(S\d+)】/g;

export default function Answer({ text, passages, cited }) {
  if (!text) return <div className="answer empty">(no answer)</div>;
  const bySid = Object.fromEntries((passages || []).map((p) => [p.sid, p]));
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(CITE_RE)) {
    parts.push(text.slice(last, m.index));
    const sid = m[1] || m[2];
    parts.push(<CiteChip key={m.index} sid={sid} passage={bySid[sid]} verified={cited.includes(sid)} />);
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return <div className="answer">{parts}</div>;
}

export function CiteChip({ sid, passage, verified = true }) {
  const title = passage
    ? `${sid} · ${passage.source}${passage.injection_markers.length ? " · FLAGGED" : ""}\n\n${passage.text}`
    : `${sid}: no retrieved passage has this id`;
  const cls = !passage ? "bad" : passage.injection_markers.length ? "flagged" : verified ? "" : "unverified";
  return (
    <span className={`cite ${cls}`} title={title} tabIndex={0} aria-label={title}>
      {sid}
    </span>
  );
}
