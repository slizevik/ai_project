"""Publish an immutable processed dataset version to S3."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import typer

from ai_project.config import DEFAULT_BUCKET, DEFAULT_VERSION, INTERIM_DATA_DIR
from ai_project.s3 import create_s3_client, object_exists

app = typer.Typer(help="Upload a processed dataset version to S3.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@app.command()
def main(
    root: Path = typer.Option(INTERIM_DATA_DIR),
    bucket: str = typer.Option(DEFAULT_BUCKET),
    version: str = typer.Option(DEFAULT_VERSION),
    validation_size: int = typer.Option(5_000),
    seed: int = typer.Option(42),
    workers: int = typer.Option(16, min=1, max=64),
) -> None:
    client = create_s3_client()
    prefix = f"processed/cifar10/{version}"
    if object_exists(client, bucket, f"{prefix}/manifest.json"):
        raise typer.BadParameter(f"Processed version already exists: {version}")

    metadata = [
        "annotations/train.json",
        "annotations/val.json",
        "annotations/test.json",
        "stats.json",
    ]
    counts: dict[str, int] = {}
    for relative in metadata:
        path = root / relative
        if not path.exists():
            raise typer.BadParameter(f"Missing pipeline artifact: {path}")
        if relative.startswith("annotations/"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            counts[Path(relative).stem] = len(payload["images"])

    manifest: dict[str, Any] = {
        "dataset": "cifar10",
        "version": version,
        "counts": counts,
        "validation_size": validation_size,
        "seed": seed,
        "git_commit": _git_commit(),
        "files": [
            {
                "path": relative,
                "size": (root / relative).stat().st_size,
                "sha256": _sha256(root / relative),
            }
            for relative in metadata
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    files = (
        [path for path in (root / "images").glob("*.png")]
        + [root / relative for relative in metadata]
        + [manifest_path]
    )

    def upload(path: Path) -> None:
        relative = path.relative_to(root).as_posix()
        client.upload_file(str(path), bucket, f"{prefix}/{relative}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(upload, files))
    typer.echo(f"Uploaded {len(files)} files to s3://{bucket}/{prefix}/")


if __name__ == "__main__":
    app()
