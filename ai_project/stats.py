from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from ai_project.config import INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(root: Path = typer.Option(INTERIM_DATA_DIR)) -> None:
    data = json.loads((root / "annotations" / "train.json").read_text(encoding="utf-8"))
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_squares = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for item in data["images"]:
        with Image.open(root / item["file_name"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flattened = pixels.reshape(-1, 3).astype(np.float64)
        channel_sum += flattened.sum(axis=0)
        channel_sum_squares += np.square(flattened).sum(axis=0)
        pixel_count += flattened.shape[0]
    mean = channel_sum / pixel_count
    variance = channel_sum_squares / pixel_count - np.square(mean)
    stats = {
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        "num_images": len(data["images"]),
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
