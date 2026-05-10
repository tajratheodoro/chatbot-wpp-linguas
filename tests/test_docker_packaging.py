from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_api_as_non_root_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim AS builder" in dockerfile
    assert "FROM python:3.12-slim AS runtime" in dockerfile
    assert "groupadd --system appuser" in dockerfile
    assert "useradd --system --gid appuser" in dockerfile
    assert "USER appuser" in dockerfile
    assert '"uvicorn", "main:app"' in dockerfile


def test_compose_uses_env_interpolation_and_pgvector_healthcheck() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "pgvector/pgvector:pg16" in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}" in compose
    assert "DATABASE_URL: ${DATABASE_URL}" in compose
    assert "condition: service_healthy" in compose
    assert "pg_isready" in compose
    assert "chatbot_pg_data:" in compose
    assert "GROQ_API_KEY" not in compose.split("api:", 1)[0]
    assert "secret" not in compose.casefold()


def test_dockerignore_excludes_secrets_and_local_artifacts() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for ignored in [".env", "tests/", "venv/", "__pycache__/", ".pytest_cache/"]:
        assert ignored in dockerignore
