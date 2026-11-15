from __future__ import annotations

import json
import random
from pathlib import Path

import typer

from ai_project.config import DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_SIZE, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    root: Path = typer.Option(INTERIM_DATA_DIR),
    validation_size: int = typer.Option(DEFAULT_VALIDATION_SIZE, min=1),
    seed: int = typer.Option(DEFAULT_SPLIT_SEED),
) -> None:
    source = json.loads((root / "annotations" / "train.json").read_text(encoding="utf-8"))
    records = source["images"]
    if validation_size >= len(records):
        raise typer.BadParameter("validation_size must be smaller than the train set")
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, object]]] = {}
    for record in records:
        by_class.setdefault(int(record["label"]), []).append(record)
    val: list[dict[str, object]] = []
    for items in by_class.values():
        rng.shuffle(items)
        val.extend(items[: round(validation_size * len(items) / len(records))])
    rng.shuffle(val)
    val = val[:validation_size]
    train = [
        item for item in records if str(item["image_id"]) not in {str(x["image_id"]) for x in val}
    ]
    for name, items in (("train", train), ("val", val)):
        (root / "annotations" / f"{name}.json").write_text(
            json.dumps({"classes": source["classes"], "images": items}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    app()
