"""Run one reproducible CPU training experiment from a YAML configuration."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import typer
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam

from ai_project.config import DEFAULT_BUCKET
from ai_project.modeling.networks import build_model
from ai_project.processed_data import load_processed_version
from ai_project.s3 import create_s3_client
from ai_project.torch_data import DataLoaders, build_dataloaders

app = typer.Typer(help="Run one CPU training experiment from a YAML config.")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, child in value.items():
            result.update(_flatten_params(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: str(value)}


def _evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    collect_predictions: bool = False,
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            losses.append(float(criterion(outputs, labels).item()))
            predicted = outputs.argmax(dim=1)
            targets.extend(labels.tolist())
            predictions.extend(predicted.tolist())
    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    return (
        float(np.mean(losses)),
        accuracy,
        macro_f1,
        targets if collect_predictions else [],
        predictions if collect_predictions else [],
    )


def _log_confusion_matrix(targets: list[int], predictions: list[int], path: Path) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(10)))
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set(title="Validation confusion matrix", xlabel="Predicted label", ylabel="True label")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _upload_artifacts(root: Path, experiment_name: str, run_name: str, bucket: str) -> None:
    client = create_s3_client()
    prefix = f"models/{experiment_name}/{run_name}"
    for path in root.iterdir():
        if path.is_file():
            client.upload_file(str(path), bucket, f"{prefix}/{path.name}")


@app.command()
def main(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Train one model and log the best validation checkpoint."""
    config_payload: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))
    experiment_name = str(config_payload["experiment_name"])
    run_name = str(config_payload["run_name"])
    seed = int(config_payload["seed"])
    data_config = config_payload["data"]
    model_config = config_payload["model"]
    train_config = config_payload["train"]
    dataset = load_processed_version(
        version=str(data_config["s3_path"]).rstrip("/").split("/")[-1],
        s3_prefix="/".join(str(data_config["s3_path"]).rstrip("/").split("/")[:-1]),
    )
    loaders: DataLoaders = build_dataloaders(
        dataset,
        batch_size=int(train_config["batch_size"]),
        seed=seed,
        max_train_samples=data_config.get("max_train_samples"),
    )
    model = build_model(str(model_config["type"]), dict(model_config.get("params", {})))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=float(train_config["lr"]))
    epochs = int(train_config["epochs"])
    best_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="cifar10-run-") as temporary_dir:
        artifact_root = Path(temporary_dir)
        checkpoint = artifact_root / "model.pt"
        confusion_path = artifact_root / "confusion_matrix.png"
        metrics_path = artifact_root / "metrics.json"
        config_path = artifact_root / "config.yaml"
        config_path.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(_flatten_params(config_payload))
            mlflow.log_param("git_commit", _git_commit())
            for epoch in range(1, epochs + 1):
                model.train()
                for inputs, labels in loaders.train:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                val_loss, val_accuracy, val_f1, _, _ = _evaluate(model, loaders.val, criterion)
                mlflow.log_metrics(
                    {"val_loss": val_loss, "val_accuracy": val_accuracy, "val_macro_f1": val_f1},
                    step=epoch,
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            val_loss, val_accuracy, val_f1, targets, predictions = _evaluate(
                model, loaders.val, criterion, collect_predictions=True
            )
            test_loss, test_accuracy, test_f1, _, _ = _evaluate(model, loaders.test, criterion)
            training_seconds = time.perf_counter() - start
            _log_confusion_matrix(targets, predictions, confusion_path)
            metrics = {
                "best_epoch": best_epoch,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_f1,
                "training_seconds": training_seconds,
                "git_commit": _git_commit(),
                "data_version": data_config["s3_path"],
                "seed": seed,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metrics(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_f1,
                    "training_seconds": training_seconds,
                }
            )
            mlflow.log_artifacts(str(artifact_root))
            _upload_artifacts(artifact_root, experiment_name, run_name, DEFAULT_BUCKET)
    typer.echo(f"Completed {experiment_name}/{run_name}; best epoch={best_epoch}")


if __name__ == "__main__":
    app()
