"""Download and resolve one immutable processed dataset version."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from ai_project.config import DEFAULT_BUCKET, EXTERNAL_DATA_DIR
from ai_project.s3 import create_s3_client, object_exists

app = typer.Typer(help="Download one processed dataset version from S3.")


@dataclass(frozen=True)
class ProcessedDataset:
    """Local paths for one self-contained processed dataset version."""

    version: str
    root: Path
    images: Path
    annotations: dict[str, Path]
    stats: Path
    manifest: Path


def _safe_relative_key(key: str, prefix: str) -> Path:
    relative = Path(key.removeprefix(f"{prefix}/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe S3 object key: {key}")
    return relative


def load_processed_version(
    version: str,
    bucket: str = DEFAULT_BUCKET,
    s3_prefix: str = "processed/cifar10",
    cache_root: Path = EXTERNAL_DATA_DIR / "cifar10",
) -> ProcessedDataset:
    """Download one version into an isolated cache directory and return its paths."""
    prefix = f"{s3_prefix.rstrip('/')}/{version}"
    root = cache_root / version
    client = create_s3_client()
    manifest_key = f"{prefix}/manifest.json"
    if not object_exists(client, bucket, manifest_key):
        raise FileNotFoundError(f"Processed version does not exist: s3://{bucket}/{prefix}/")

    root.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        objects.extend(page.get("Contents", []))
    if not objects:
        raise FileNotFoundError(f"Processed version is empty: s3://{bucket}/{prefix}/")

    for item in objects:
        key = str(item["Key"])
        relative = _safe_relative_key(key, prefix)
        destination = root / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(destination))

    annotations = {
        name: root / "annotations" / f"{name}.json" for name in ("train", "val", "test")
    }
    stats = root / "stats.json"
    manifest = root / "manifest.json"
    required = [*annotations.values(), stats, manifest]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Processed version is incomplete; missing: {missing}")

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("version") != version:
        raise ValueError(f"Manifest version does not match requested version: {version}")
    return ProcessedDataset(version, root, root / "images", annotations, stats, manifest)


@app.command()
def main(
    version: str = typer.Argument(..., help="Processed dataset version, for example v1."),
    bucket: str = typer.Option(DEFAULT_BUCKET),
    cache_root: Path = typer.Option(EXTERNAL_DATA_DIR / "cifar10"),
) -> None:
    dataset = load_processed_version(version, bucket=bucket, cache_root=cache_root)
    typer.echo(f"Cached {dataset.version} at {dataset.root}")


if __name__ == "__main__":
    app()
