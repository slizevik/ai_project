"""PyTorch datasets and deterministic DataLoader construction for CIFAR-10."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ai_project.processed_data import ProcessedDataset


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible data loading and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CifarImageDataset(Dataset[tuple[torch.Tensor, int]]):  # type: ignore[misc]
    """Read PNG images and labels from one processed annotation file."""

    def __init__(
        self,
        root: Path,
        annotation_path: Path,
        mean: list[float],
        std: list[float],
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        payload: dict[str, Any] = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = list(payload["images"])
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            if max_samples < len(records):
                random.Random(seed).shuffle(records)
                records = records[:max_samples]
        self.root = root
        self.records = records
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = self.root / str(record["file_name"])
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor, int(record["label"])


@dataclass(frozen=True)
class DataLoaders:
    """Train, validation and test DataLoaders for one dataset version."""

    train: DataLoader[tuple[torch.Tensor, int]]
    val: DataLoader[tuple[torch.Tensor, int]]
    test: DataLoader[tuple[torch.Tensor, int]]


def _read_stats(path: Path) -> tuple[list[float], list[float]]:
    stats: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    mean = [float(value) for value in stats["mean"]]
    std = [float(value) for value in stats["std"]]
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("stats.json must contain three positive standard deviations")
    return mean, std


def build_dataloaders(
    dataset: ProcessedDataset,
    batch_size: int,
    seed: int,
    max_train_samples: int | None = None,
) -> DataLoaders:
    """Build deterministic train/val/test loaders using one version's statistics."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    seed_everything(seed)
    mean, std = _read_stats(dataset.stats)
    train = CifarImageDataset(
        dataset.root,
        dataset.annotations["train"],
        mean,
        std,
        max_samples=max_train_samples,
        seed=seed,
    )
    val = CifarImageDataset(dataset.root, dataset.annotations["val"], mean, std)
    test = CifarImageDataset(dataset.root, dataset.annotations["test"], mean, std)
    generator = torch.Generator().manual_seed(seed)
    return DataLoaders(
        train=DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        val=DataLoader(val, batch_size=batch_size, shuffle=False),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )
