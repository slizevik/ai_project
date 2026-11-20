"""Shared S3 client configuration and small object helpers."""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.client import Config

from ai_project.config import (
    AWS_REQUEST_CHECKSUM_CALCULATION,
    AWS_RESPONSE_CHECKSUM_VALIDATION,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_S3_REGION,
    S3_ACCESS_KEY,
    S3_ADDRESSING_STYLE,
    S3_SECRET_KEY,
)


def create_s3_client() -> Any:
    """Create a boto3 client configured for local or remote S3-compatible storage."""
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", AWS_REQUEST_CHECKSUM_CALCULATION)
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", AWS_RESPONSE_CHECKSUM_VALIDATION)
    return boto3.client(
        "s3",
        endpoint_url=DEFAULT_S3_ENDPOINT,
        region_name=DEFAULT_S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(s3={"addressing_style": S3_ADDRESSING_STYLE}),
    )


def object_exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether an object exists, treating a missing key as a normal result."""
    try:
        client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True
