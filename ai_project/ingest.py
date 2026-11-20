"""Download the official CIFAR-10 archive and store it unchanged in S3."""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import typer
from loguru import logger

from ai_project.config import CIFAR10_URL, DEFAULT_BUCKET, RAW_OBJECT_KEY
from ai_project.s3 import create_s3_client, object_exists

app = typer.Typer(help="Upload the immutable CIFAR-10 raw archive to S3.")


@app.command()
def main(
    bucket: str = typer.Option(DEFAULT_BUCKET, help="Destination S3 bucket."),
    key: str = typer.Option(RAW_OBJECT_KEY, help="Destination object key."),
    url: str = typer.Option(CIFAR10_URL, help="Official CIFAR-10 archive URL."),
) -> None:
    """Download and upload the archive once; never overwrite an existing object."""
    client = create_s3_client()
    if object_exists(client, bucket, key):
        logger.info("Raw object already exists: s3://{}/{}", bucket, key)
        return

    with tempfile.TemporaryDirectory(prefix="cifar10-ingest-") as temporary_dir:
        archive_path = Path(temporary_dir) / "cifar-10-python.tar.gz"
        logger.info("Downloading CIFAR-10 archive from {}", url)
        urllib.request.urlretrieve(url, archive_path)
        logger.info("Uploading immutable raw object to s3://{}/{}", bucket, key)
        client.upload_file(str(archive_path), bucket, key)
    logger.success("Raw CIFAR-10 archive uploaded")


if __name__ == "__main__":
    app()
