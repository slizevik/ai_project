FROM python:3.11.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY ai_project ./ai_project
COPY configs ./configs

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "ai_project.run_grid", "configs/grid.yaml"]
