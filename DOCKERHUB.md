# route-or-roam

The same grounded Q&A task built twice — a fixed LangGraph workflow and a free
ReAct agent — over the same tools, budget and verifier, with a FastAPI API and
a React UI on port 8001. Source, results and method:
https://github.com/Siegried81/route-or-roam

```bash
docker run -d -p 8001:8001 --env-file .env siegried/route-or-roam
# open http://localhost:8001   (health: /api/health)
```

Answering needs grounded-rag's indexes and an LLM key (`GROQ_API_KEY`, or a
local Ollama via `OLLAMA_URL`). The indexes are built once with the `ingest`
service of the repository's `docker-compose.yml`, or on start with
`RR_INGEST_ON_START=1`. Without them the API and UI still start.

Tags: `latest` (main), `sha-<commit>`, and `<version>` for release tags.
