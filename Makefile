.PHONY: setup check s3-up s3-down ingest data infra-up experiments infra-down

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

infra-up:
	docker compose build training
	docker compose up -d rustfs s3-init postgres mlflow

experiments:
	@set -e; for index in 0 1 2 3 4 5 6; do \
		echo "Starting experiment container $$index"; \
		docker compose --profile training run --rm \
			-e CONFIG=$(or $(CONFIG),configs/grid.yaml) \
			-e GRID_INDEX=$$index training; \
	done

infra-down:
	docker compose down
