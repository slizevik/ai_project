from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import typer
from PIL import Image

from ai_project.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _read_batch(member: tarfile.TarFile, name: str) -> tuple[np.ndarray, list[int]]:
    extracted = member.extractfile(name)
    if extracted is None:
        raise ValueError(f"Missing archive member: {name}")
    data: dict[bytes, Any] = pickle.load(extracted, encoding="bytes")
    return np.asarray(data[b"data"], dtype=np.uint8), [int(x) for x in data[b"labels"]]


@app.command()
def main(
    archive: Path = typer.Option(RAW_DATA_DIR / "cifar-10-python.tar.gz"),
    output: Path = typer.Option(INTERIM_DATA_DIR),
) -> None:
    images_dir = output / "images"
    annotations_dir = output / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    annotations: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    with tarfile.open(archive, "r:gz") as tar:
        for split, members in (
            ("train", [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]),
            ("test", ["cifar-10-batches-py/test_batch"]),
        ):
            index = 0
            for member_name in members:
                pixels, labels = _read_batch(tar, member_name)
                for row, label in zip(pixels, labels, strict=False):
                    image_id = f"{split}_{index:05d}"
                    path = images_dir / f"{image_id}.png"
                    Image.fromarray(row.reshape(3, 32, 32).transpose(1, 2, 0)).save(
                        path, format="PNG"
                    )
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    annotations[split].append(
                        {
                            "image_id": image_id,
                            "file_name": f"images/{path.name}",
                            "label": label,
                            "label_name": CLASSES[label],
                            "sha256": digest,
                        }
                    )
                    index += 1
    for split, records in annotations.items():
        (annotations_dir / f"{split}.json").write_text(
            json.dumps({"classes": CLASSES, "images": records}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    app()
