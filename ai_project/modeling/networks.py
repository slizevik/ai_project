"""Small CPU-friendly CIFAR-10 models and a configuration-based factory."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LinearBaseline(nn.Module):
    """A single linear classifier over flattened 32x32 RGB pixels."""

    def __init__(self, input_size: int = 3 * 32 * 32, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.flatten(start_dim=1))


class SimpleCNN(nn.Module):
    """A configurable 2–3 layer convolutional classifier for CIFAR-10."""

    def __init__(
        self, channels: list[int] | tuple[int, ...] = (32, 64), num_classes: int = 10
    ) -> None:
        super().__init__()
        if len(channels) not in {2, 3} or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain two or three positive values")
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(model_type: str, params: dict[str, Any]) -> nn.Module:
    """Build a model from the YAML model section without changing training code."""
    if model_type == "linear":
        return LinearBaseline(
            input_size=int(params.get("input_size", 3 * 32 * 32)),
            num_classes=int(params.get("num_classes", 10)),
        )
    if model_type == "simple_cnn":
        raw_channels = params.get("channels", [32, 64])
        channels = tuple(int(channel) for channel in raw_channels)
        return SimpleCNN(channels=channels, num_classes=int(params.get("num_classes", 10)))
    raise ValueError(f"Unknown model type: {model_type}")
