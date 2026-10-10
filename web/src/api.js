// Tiny fetch wrapper: JSON in, JSON out, FastAPI error details surfaced as Error messages.
// The HTTP status is kept on the error (err.status) so callers can tell "nothing to
// resume" (404) or "already resuming" (409) from a transient failure worth retrying.
// The approval token, when the server requires one (RR_APPROVE_TOKEN set).
// Kept in localStorage so it is typed once per browser, never in the bundle.
const TOKEN_KEY = "rr_approve_token";
export const approveToken = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } },
  set: (value) => { try { value ? localStorage.setItem(TOKEN_KEY, value) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ } },
};

export async function api(path, body) {
  const headers = { "Content-Type": "application/json" };
  // Only the approval call writes anything, so only it carries the token.
  if (path === "/api/approve" && approveToken.get()) headers["X-Approve-Token"] = approveToken.get();
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ")
      : detail || `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

export const fmt = {
  pct: (v, d = 1) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(d)}%`),
  // Run-to-run spread; null with a single repeat, where no spread is observable.
  spread: (v) => (v === null || v === undefined ? "± n/a" : `± ${(100 * v).toFixed(1)}`),
  ci: (w) => (w ? `${Math.round(100 * w[0])}–${Math.round(100 * w[1])}%` : "n/a"),
  pctRaw: (v, d = 1) => (v === null || v === undefined ? "n/a" : `${v.toFixed(d)}%`),
  num: (v, d = 1) => (v === null || v === undefined ? "n/a" : v.toFixed(d)),
  int: (v) => (v === null || v === undefined ? "n/a" : Math.round(v).toLocaleString("en-US")),
  sec: (v) => (v === null || v === undefined ? "n/a" : `${v.toFixed(1)}s`),
};

export const SYSTEM_LABEL = { workflow: "System A · workflow", agent: "System B · agent" };
