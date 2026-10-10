#!/bin/sh
# Start the route-or-roam API, after building grounded-rag's indexes if asked.
#
# RR_INGEST_ON_START=1 runs grounded-rag's ingest for each corpus in
# RR_CORPORA (default: both shipped corpora) whose index is missing. It needs
# the embedder grounded-rag is configured with (EMBED_PROVIDER: ollama, hosted
# or sentence_transformers) to be reachable, and it is what a platform with no
# persistent disk (Render) uses on every deploy. With a persistent index
# volume (docker compose) leave it unset and run the `ingest` service once.
set -eu

: "${GROUNDED_RAG_PATH:=/opt/grounded-rag}"
: "${PORT:=8001}"
: "${RR_CORPORA:=ai_act_sections filings_sections}"

if [ "${RR_INGEST_ON_START:-0}" = "1" ]; then
  for corpus in $RR_CORPORA; do
    if [ ! -d "$GROUNDED_RAG_PATH/index/$corpus" ]; then
      echo "==> building grounded-rag index for $corpus"
      (cd "$GROUNDED_RAG_PATH" && python cli.py ingest --corpus "$corpus")
    fi
  done
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
