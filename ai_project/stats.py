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
    arrays = [
        np.asarray(Image.open(root / item["file_name"]), dtype=np.float64) / 255.0
        for item in data["images"]
    ]
    pixels = np.concatenate([image.reshape(-1, 3) for image in arrays])
    stats = {
        "mean": pixels.mean(axis=0).tolist(),
        "std": pixels.std(axis=0).tolist(),
        "num_images": len(arrays),
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
