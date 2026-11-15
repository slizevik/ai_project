from __future__ import annotations

from pathlib import Path

import typer

from ai_project.config import (
    DEFAULT_BUCKET,
    DEFAULT_VERSION,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    RAW_OBJECT_KEY,
)
from ai_project.convert import main as convert_main
from ai_project.download_raw import main as download_main
from ai_project.split import main as split_main
from ai_project.stats import main as stats_main
from ai_project.upload_processed import main as upload_main

app = typer.Typer(help="Run the local CIFAR-10 preparation pipeline.")


@app.command()
def main(
    raw_archive: Path = typer.Option(RAW_DATA_DIR / "cifar-10-python.tar.gz"),
    output: Path = typer.Option(INTERIM_DATA_DIR),
    validation_size: int = typer.Option(5_000, min=1),
    seed: int = typer.Option(42),
    version: str = typer.Option(DEFAULT_VERSION),
) -> None:
    download_main(bucket=DEFAULT_BUCKET, key=RAW_OBJECT_KEY, output=raw_archive)
    convert_main(archive=raw_archive, output=output)
    split_main(root=output, validation_size=validation_size, seed=seed)
    stats_main(root=output)
    upload_main(
        root=output,
        bucket=DEFAULT_BUCKET,
        version=version,
        validation_size=validation_size,
        seed=seed,
    )
    typer.echo(f"Pipeline complete: {output}")


if __name__ == "__main__":
    app()
