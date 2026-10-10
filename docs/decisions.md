# Decisions

One dated entry per change that alters what the system promises. Append, never rewrite.

## 2026-10-10 — Approving over the API needs an identity

- **What:** `POST /api/approve` checks who calls it. With `RR_APPROVE_TOKEN`
  unset, only requests from this machine may approve (403 otherwise); with it
  set, the `X-Approve-Token` header must match it, compared in constant time
  (401 otherwise). The response carries `decided_by`. The three other routes
  are unchanged: they read, approving writes.
- **Why:** the human gate is the agent's one irreversible action, and the API
  accepted the decision from anyone who could reach the port. On localhost
  that is the owner; started with `--host 0.0.0.0`, it was anyone on the
  network, writing a report in the owner's name.
- **What it does not settle:** a token is one shared secret, not a user. If
  the API is ever exposed to several people, replace it with a session and a
  name in `decided_by`.
- **Revisit if:** the React UI is deployed off-box — it then has to send the
  header (`web/src/AskTab.jsx`, the `api("/api/approve", …)` call).
