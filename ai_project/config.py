"""Project configuration and defaults for the CIFAR-10 data pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
RAW_OBJECT_KEY = "raw/cifar-10-python.tar.gz"
DEFAULT_BUCKET = os.getenv("S3_BUCKET", "ai-project")
DEFAULT_S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
DEFAULT_S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ADDRESSING_STYLE = os.getenv("S3_ADDRESSING_STYLE", "path")
DEFAULT_VALIDATION_SIZE = 5_000
DEFAULT_SPLIT_SEED = 42
DEFAULT_VERSION = "v1"

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "rustfsadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "rustfsadmin")

# boto3/botocore only calculate and validate request checksums when required.
# This keeps local S3-compatible backends compatible with current boto3.
AWS_REQUEST_CHECKSUM_CALCULATION = os.getenv("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
AWS_RESPONSE_CHECKSUM_VALIDATION = os.getenv("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

logger.info("PROJ_ROOT path is: {}", PROJ_ROOT)
