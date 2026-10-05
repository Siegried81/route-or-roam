import { useState } from "react";
import { fmt } from "../api.js";

const SERIES = [
  { id: "workflow", label: "Workflow", color: "var(--series-1)" },
  { id: "agent", label: "Agent", color: "var(--series-2)" },
];

// Bar with 4px rounded top corners anchored to the baseline.
function barPath(x, y, w, h, r = 4) {
  if (h <= 0) return "";
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`;
}

// Grouped bars of success rate by question type, with +/- sample std whiskers
// (run-to-run spread across repeats); one fixed colour per system.
export default function BarChart({ data }) {
  const [hover, setHover] = useState(null);
  const types = data.type_order;
  const series = SERIES.filter((s) => data.systems[s.id]);
  const W = 720, H = 300;
  const m = { t: 16, r: 12, b: 44, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const band = iw / Math.max(1, types.length);
  const gap = 2;
  const barW = Math.min(44, (band * 0.7 - gap * (series.length - 1)) / Math.max(1, series.length));
  const groupW = barW * series.length + gap * (series.length - 1);
  const y = (v) => m.t + ih - v * ih;

  const bars = [];
  types.forEach((t, ti) => {
    series.forEach((s, si) => {
      const st = data.systems[s.id].by_type[t];
      if (!st || st.mean === null) return;
      const x = m.l + ti * band + (band - groupW) / 2 + si * (barW + gap);
      bars.push({ key: `${t}-${s.id}`, t, s, st, x });
    });
  });

  const summary = bars.map((b) => `${b.t} ${b.s.label} ${fmt.pct(b.st.mean, 0)}`).join(", ");

  return (
    <figure className="chart">
      <div className="legend" aria-hidden="true">
        {series.map((s) => (
          <span key={s.id}><span className="legend-swatch" style={{ background: s.color }} />{s.label}</span>
        ))}
        <span className="muted">whiskers: ± std across repeats</span>
      </div>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Success by question type. ${summary}`}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} className={v === 0 ? "axis" : "grid"} />
              <text x={m.l - 8} y={y(v)} dy="0.32em" textAnchor="end" className="tick">{v * 100}%</text>
            </g>
          ))}
          {types.map((t, ti) => (
            <text key={t} x={m.l + ti * band + band / 2} y={H - m.b + 18} textAnchor="middle" className="tick">
              {t.replace("_", " ")}
            </text>
          ))}
          {bars.map((b) => {
            const top = y(b.st.mean);
            const sd = b.st.spread || 0;
            const cx = b.x + barW / 2;
            const hi = y(Math.min(1, b.st.mean + sd));
            const lo = y(Math.max(0, b.st.mean - sd));
            const active = hover && hover.key === b.key;
            return (
              <g key={b.key} onMouseEnter={() => setHover(b)} onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(b)} onBlur={() => setHover(null)} tabIndex={0}
                aria-label={`${b.t}, ${b.s.label}: ${fmt.pct(b.st.mean)} plus or minus ${fmt.pct(sd)}`}
                className={`bar ${hover && !active ? "dim" : ""}`}>
                <rect x={b.x - gap} y={m.t} width={barW + 2 * gap} height={ih} fill="transparent" />
                <path d={barPath(b.x, top, barW, m.t + ih - top)} fill={b.s.color} />
                {b.st.mean === 0 && (
                  <line x1={b.x} x2={b.x + barW} y1={y(0) - 1} y2={y(0) - 1} stroke={b.s.color} strokeWidth="2" />
                )}
                {sd > 0 && (
                  <g className="whisker">
                    <line x1={cx} x2={cx} y1={hi} y2={lo} />
                    <line x1={cx - 5} x2={cx + 5} y1={hi} y2={hi} />
                    <line x1={cx - 5} x2={cx + 5} y1={lo} y2={lo} />
                  </g>
                )}
              </g>
            );
          })}
        </svg>
        {hover && (
          <div className="tooltip" role="status"
            style={{ left: `${((hover.x + barW / 2) / W) * 100}%`, top: `${(y(hover.st.mean) / H) * 100}%` }}>
            <div className="tt-title">{hover.t.replace("_", " ")}</div>
            <div>
              <span className="legend-swatch" style={{ background: hover.s.color }} />{hover.s.label}
              <strong> {fmt.pct(hover.st.mean)}</strong>
            </div>
            <div className="muted">
              {fmt.spread(hover.st.spread)} over {hover.st.repeats} repeat(s),
              n={data.systems[hover.s.id].n_by_type?.[hover.t]}
            </div>
          </div>
        )}
      </div>
    </figure>
  );
}
