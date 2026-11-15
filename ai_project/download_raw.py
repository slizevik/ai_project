from __future__ import annotations

from pathlib import Path

import typer

from ai_project.config import DEFAULT_BUCKET, RAW_DATA_DIR, RAW_OBJECT_KEY
from ai_project.s3 import create_s3_client

app = typer.Typer()


@app.command()
def main(
    bucket: str = typer.Option(DEFAULT_BUCKET),
    key: str = typer.Option(RAW_OBJECT_KEY),
    output: Path = typer.Option(RAW_DATA_DIR / "cifar-10-python.tar.gz"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    create_s3_client().download_file(bucket, key, str(output))
    typer.echo(f"Downloaded {key} to {output}")


if __name__ == "__main__":
    app()
