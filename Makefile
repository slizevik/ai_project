.PHONY: setup check s3-up s3-down ingest data

setup:
	uv sync --locked --all-groups
	uv run --locked pre-commit install

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy

s3-up:
	docker compose up -d rustfs s3-init

s3-down:
	docker compose down

ingest:
	uv run --locked python -m ai_project.ingest

data:
	uv run --locked python -m ai_project.pipeline
