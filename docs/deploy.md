# CI and deployment

One page for everything about shipping the image: CI, the Docker image,
Render, Docker Hub and Streamlit Community Cloud. The Hub page text itself is
`DOCKERHUB.md` (the publish workflow uploads it).

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request, in three jobs:

- **tests**: the offline suite (`python -m pytest -q`), with grounded-rag
  checked out next to the repo at a fixed commit (`GROUNDED_RAG_REF` in the
  workflow). That commit is the one `eval/questions.jsonl` is labelled
  against, and the `Dockerfile` builds the same one (a test checks they
  agree). To bump both: check out the new commit, run
  `python -m eval.locate_gold`, relabel what it reports, then change the sha.
- **web**: `npm ci && npm run build` on Node 26, the version of the image.
- **docker**: builds the image, starts it with the fake embedder
  (`RR_INGEST_ON_START=1 EMBED_PROVIDER=fake`, no key) and waits for
  `/api/health`.

Actions are pinned to commit SHAs. Dependabot (`.github/dependabot.yml`) opens
one grouped PR a month per ecosystem (pip, npm, Actions, base images); majors
for npm and the base images come in a second grouped PR.
`tests/test_deploy_config.py` keeps `Dockerfile`, `docker-compose.yml`,
`render.yaml` and the workflow in step with the code: health route, port,
corpora, `RR_INGEST_ON_START` on Render, no secret value committed.

Python versions: `requirements.txt` holds the supported floors,
`requirements.lock` the exact versions the image ships (grounded-rag's
requirements and this project's, resolved together for Python 3.12 on Linux),
and `requirements-dev.txt` the test tools. After a floor changes, regenerate
the lock with the command at its top; `tests/test_deploy_config.py` fails
until the lock satisfies every floor.

## Docker

grounded-rag is imported in place (its `rag` package, `config`, `data/` and
`index/`), so a deployment ships a checkout of it, not a pip package. The
`Dockerfile` clones it at build time at a pinned commit (`--build-arg
GROUNDED_RAG_REF=<sha>` to override), builds the UI in a Node stage and starts
uvicorn on `$PORT`.

```bash
docker compose --profile local-llm up -d ollama                      # bundled embedder
docker compose --profile local-llm exec ollama ollama pull nomic-embed-text
docker compose --profile tools --profile local-llm run --rm ingest   # both indexes, once, into the rag_index volume
docker compose up --build                                            # API + UI on :8001
```

The published image runs alone with `docker run -d -p 8001:8001 --env-file .env
siegried/route-or-roam`. Without grounded-rag's indexes the API and UI start
but cannot answer: build them with the `ingest` service above, or set
`RR_INGEST_ON_START=1` to build them at start.

Which LLM answers: Groq with `GROQ_API_KEY` (up to five keys), or a local
Ollama with `RR_LLM_PROVIDER=ollama` and `RR_OLLAMA_BASE_URL`. `OLLAMA_URL` is
a different thing: the embedder of grounded-rag.

`/api/ask` spends Groq tokens on every call, so a public deploy needs a guard.
Both are off by default: `RR_RATE_LIMIT_PER_MINUTE` caps asks per client IP
over a sliding minute (`render.yaml` sets 6), and `RR_API_KEY` makes callers
send `X-API-Key` (API-only deploys: the bundled UI does not send one).
`RR_APPROVE_TOKEN` is the secret for `POST /api/approve`.

## Render

`render.yaml` is a Render blueprint for the same image. There is no Ollama
there, so it sets `EMBED_PROVIDER=hosted` and `RR_INGEST_ON_START=1`, which
builds the indexes on every start with the hosted embedder named in the
dashboard (`HOSTED_EMBED_BASE_URL`, `HOSTED_EMBED_MODEL`,
`HOSTED_EMBED_API_KEY`), plus `GROQ_API_KEY`. `RR_APPROVE_TOKEN` is generated
by Render and typed once into the approval dialog. CI builds and smoke-tests
this image; the Render deploy itself is not exercised, so the first deploy is
its test.

## Streamlit Community Cloud

It clones only this repository, so the checkout and the indexes are made at
first import instead. Deploy `app.py` and put these in the app's *Secrets*
(Streamlit exports them as environment variables):

```toml
RR_FETCH_GROUNDED_RAG = "1"          # download grounded-rag's archive into GROUNDED_RAG_PATH
GROUNDED_RAG_PATH = ".grounded-rag"  # relative to this repository
GROUNDED_RAG_REF = "main"            # branch, tag or commit sha
RR_INGEST_ON_START = "1"             # build a corpus index on first use
EMBED_PROVIDER = "hosted"            # no Ollama there; any OpenAI-compatible /embeddings endpoint
HOSTED_EMBED_BASE_URL = "https://openrouter.ai/api/v1"
HOSTED_EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"
HOSTED_EMBED_API_KEY = "..."
GROQ_API_KEY = "..."
```

The platform has no persistent disk, so both happen again on every cold start,
and the first question after one waits for the download (a few MB) and the
embedding of that corpus. `requirements.txt` already carries what grounded-rag's
retrieval imports; its PDF reader is not needed, the shipped corpora are text.
