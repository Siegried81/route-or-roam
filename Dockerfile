# route-or-roam: the API (port 8001) serving the built React UI, with grounded-rag inside.
#
# Why grounded-rag is cloned into the image: rr/grounded.py imports it in place
# (its `rag` package, its `config` module, its data/ corpora and its index/),
# so a deployment needs a checkout of it next to this code, not a pip package.
# The ref is pinned by build argument so an image is reproducible; bump it on
# purpose, after re-running the evaluation.
#
# Three stages: the UI is built with Node and only web/dist is kept; Python
# dependencies are installed before the source so code edits rebuild one cheap
# layer; the runtime image carries no compiler, no Node and no .env.

# --- 1. React UI ------------------------------------------------------------
FROM node:25-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --silent 2>/dev/null || npm install --silent
COPY web/ ./
RUN npm run build

# --- 2. Python runtime ------------------------------------------------------
FROM python:3.12-slim

ARG GROUNDED_RAG_REPO=https://github.com/Siegried81/grounded-rag
ARG GROUNDED_RAG_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GROUNDED_RAG_PATH=/opt/grounded-rag

# git only for the checkout; removed in the same layer so it is not in the image.
# fetch + checkout FETCH_HEAD accepts a branch, a tag or a commit sha alike.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git init -q /opt/grounded-rag \
    && git -C /opt/grounded-rag fetch -q --depth 1 "${GROUNDED_RAG_REPO}" "${GROUNDED_RAG_REF}" \
    && git -C /opt/grounded-rag checkout -q FETCH_HEAD \
    && rm -rf /opt/grounded-rag/.git \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Both dependency sets, grounded-rag's first: this project's requirements list
# only what it adds on top.
RUN pip install -r /opt/grounded-rag/requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged user. reports/, runs/ and grounded-rag's index/ are written at
# runtime and live in volumes (see docker-compose.yml).
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/reports /app/runs /opt/grounded-rag/index \
    && chown -R app:app /app /opt/grounded-rag

COPY --chown=app:app rr/ ./rr/
COPY --chown=app:app api/ ./api/
COPY --chown=app:app eval/ ./eval/
COPY --chown=app:app app.py cli.py ./
COPY --chown=app:app docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --from=web --chown=app:app /web/dist ./web/dist

USER app
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; p=os.environ.get('PORT','8001'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=4).status == 200 else 1)"]

# The entrypoint builds grounded-rag's indexes when asked and missing, then
# starts uvicorn on $PORT (Render sets it; 8001 otherwise).
ENTRYPOINT ["entrypoint.sh"]
