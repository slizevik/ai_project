"""Run a baseline and a Cartesian hyperparameter grid from YAML."""

from __future__ import annotations

import copy
import itertools
import tempfile
from pathlib import Path
from typing import Any

import typer
import yaml  # type: ignore

from ai_project.train import main as train_main

app = typer.Typer(help="Run baseline and grid experiments from a YAML file.")


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Grid key does not point to a mapping: {dotted_key}")
        target = child
    target[parts[-1]] = value


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _run_config(payload: dict[str, Any], label: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="cifar10-", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        path = Path(handle.name)
    try:
        typer.echo(f"Starting {label}: {payload['run_name']}")
        train_main(path, allow_overwrite=False)
    finally:
        path.unlink(missing_ok=True)


@app.command()
def main(
    config: Path = typer.Argument(..., exists=True, readable=True),
    index: int | None = typer.Option(
        None,
        "--index",
        min=0,
        help="Run only one item: 0 is baseline, 1-6 are grid combinations.",
    ),
) -> None:
    """Run the baseline first, then every combination in the configured grid."""
    grid_payload = _load(config)
    config_root = config.parent

    def resolve_config(raw_path: str) -> Path:
        candidates = [config_root / raw_path, config_root.parent / raw_path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not find config: {raw_path}")

    baseline_path = resolve_config(str(grid_payload.get("baseline", "baseline.yaml")))
    base_path = resolve_config(str(grid_payload["base"]))
    baseline = _load(baseline_path)
    base = _load(base_path)
    if index is None or index == 0:
        _run_config(baseline, "baseline")
    elif index < 0:
        raise typer.BadParameter("index must be non-negative")

    grid = grid_payload.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid.yaml must contain a non-empty grid mapping")
    keys = list(grid)
    values = [grid[key] for key in keys]
    for combination_index, combination in enumerate(itertools.product(*values), start=1):
        if index is not None and index != combination_index:
            continue
        experiment = copy.deepcopy(base)
        for key, value in zip(keys, combination, strict=True):
            _set_nested(experiment, key, value)
        experiment["run_name"] = (
            f"cnn-lr{experiment['train']['lr']}-bs{experiment['train']['batch_size']}"
            f"-seed{experiment['seed']}"
        )
        _run_config(experiment, f"grid-{combination_index}")


if __name__ == "__main__":
    app()
