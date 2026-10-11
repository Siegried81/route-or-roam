"""The deployment files describe the same app as the code: routes, port, corpora, secrets.

Dockerfile, docker-compose.yml, render.yaml and the CI workflow are never
imported by the app, so without a test a renamed route, a corpus added to
rr/settings.py or a dropped RR_INGEST_ON_START would break the image or the
Render service silently. Every check here is offline and takes milliseconds.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from api.main import app
from rr import settings

ROOT = Path(__file__).resolve().parent.parent
API_ROUTES = {r.path for r in app.routes if getattr(r, "path", "").startswith("/api")}


def _yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_render_blueprint_matches_the_app():
    """One Docker web service, health-checked on a real route, indexes built at start, no secret value."""
    (service,) = _yaml("render.yaml")["services"]
    assert service["runtime"] == "docker"
    assert (ROOT / service["dockerfilePath"]).is_file()
    assert service["healthCheckPath"] in API_ROUTES
    env = {var["key"]: var for var in service["envVars"]}
    # Free plan, no disk: without this the service starts with nothing to search.
    assert env["RR_INGEST_ON_START"]["value"] == "1"
    for var in env.values():
        # sync: false means "set it in the dashboard"; such a variable must carry no value.
        assert not (var.get("sync") is False and "value" in var), f"{var['key']} would leak a secret"


def test_dockerfile_runs_unprivileged_with_a_real_healthcheck():
    text = _dockerfile()
    assert "USER app" in text, "the server must not run as root"
    assert "HEALTHCHECK" in text and "/api/health" in text and "/api/health" in API_ROUTES
    assert 'ENTRYPOINT ["entrypoint.sh"]' in text and "docker/entrypoint.sh" in text
    assert (ROOT / "docker" / "entrypoint.sh").is_file()


def test_entrypoint_builds_every_configured_corpus():
    """A corpus added to rr/settings.py must also be indexed at container start-up."""
    script = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    match = re.search(r"RR_CORPORA:=([^}]+)\}", script)
    assert match, "entrypoint.sh must declare a default for RR_CORPORA"
    assert set(match.group(1).split()) == set(settings.CORPORA)


def test_compose_persists_the_index_where_the_image_expects_it():
    compose = _yaml("docker-compose.yml")
    index_dir = re.search(r"GROUNDED_RAG_PATH=(\S+)", _dockerfile()).group(1) + "/index"
    api = compose["services"]["api"]
    assert index_dir in [mount.split(":")[1] for mount in api["volumes"]]
    assert "8001:8001" in api["ports"]


def test_ci_workflow_covers_tests_web_and_docker():
    workflow = _yaml(".github/workflows/ci.yml")
    jobs = workflow["jobs"]
    # The tests run against a fixed grounded-rag commit, so a change there cannot
    # turn this repo's CI red on its own (see the comment in the workflow).
    assert re.fullmatch(r"[0-9a-f]{40}", workflow["env"]["GROUNDED_RAG_REF"])
    assert {"tests", "web", "docker"} <= jobs.keys()
    steps = " ".join(str(step) for step in jobs["tests"]["steps"])
    assert "grounded-rag" in steps, "the tests need grounded-rag checked out next to the repo"
