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

## 2026-10-10 — Deployable as one Docker image, grounded-rag cloned inside

- **What:** `Dockerfile` (Node stage for the UI, Python runtime with grounded-rag
  cloned at a pinned ref into `/opt/grounded-rag`), `docker/entrypoint.sh`
  (optional index build at start, uvicorn on `$PORT`), `docker-compose.yml`
  (API, one-shot `ingest`, bundled Ollama, `rag_index` volume), `render.yaml`
  (hosted embedder, indexes rebuilt on each start). The UI sends
  `X-Approve-Token` from a field that appears only once the server asks for it.
- **Why:** route-or-roam reaches grounded-rag through `sys.path` and uses its
  corpora and indexes, so "install the package" is not a deployment; a checkout
  at a known ref is. Without the header the identity check added the same day
  would have locked the deployed UI out of approving.
- **What it does not settle:** the Render blueprint is written from the
  platform's documented behaviour, not from a deploy; the first deploy is the
  test. Rebuilding indexes on every start costs embedding calls each deploy,
  acceptable for two small corpora, not for a large one.
- **Revisit if:** grounded-rag gains a `pyproject.toml` and ships its corpora
  as package data — the clone then becomes a pip install at a version.
