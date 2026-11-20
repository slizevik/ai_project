# AGENTS.md

## Homework 2 implementation strategy

Build the data workflow in layers so local data processing can be validated independently from S3:

1. Create one reliable, typed S3 client/configuration layer.
2. Implement the local pipeline with explicit intermediate artifacts and independent CLIs:
   - download raw archive from S3;
   - convert CIFAR-10 pickle data to PNG images and JSON annotations;
   - create a deterministic stratified train/validation split;
   - calculate train-only RGB statistics;
   - create a manifest and upload an immutable processed version.
3. Add Makefile orchestration and Docker Compose/RustFS only after the client and local pipeline contracts are stable.
4. Verify idempotency, immutability, reproducibility, and `make check` before delivery.

This ordering keeps infrastructure failures separate from data-processing failures and allows each step to be run independently.

## Fixed defaults

- Dataset: CIFAR-10 Python archive from the official Toronto source.
- Validation size: 5,000 images selected from the original 50,000-image train set.
- Split seed: 42.
- First processed version: `v1`.
- Object layout:
  - raw archive: `raw/cifar-10-python.tar.gz`
  - processed prefix: `processed/cifar10/<version>/`
- Local intermediate roots: `data/raw/`, `data/interim/`, and `data/processed/`.

All operational values must remain configurable through CLI arguments or environment variables; these are defaults, not hard-coded execution constraints.

## Module boundaries

- `ai_project/s3.py`: environment-backed S3 client and object helpers.
- `ai_project/ingest.py`: download the official archive and upload it once.
- `ai_project/download_raw.py`: download the raw S3 object locally.
- `ai_project/convert.py`: trusted pickle to PNG/JSON conversion.
- `ai_project/split.py`: deterministic stratified split.
- `ai_project/stats.py`: train-only channel statistics.
- `ai_project/upload_processed.py`: immutable versioned upload and manifest.
- `ai_project/pipeline.py`: one-command orchestration of the independent steps.

Do not commit generated datasets, credentials, or local `.env` files.
