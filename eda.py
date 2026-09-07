"""Optional visual exploration for the housing dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import TARGET


def save_plots(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save the core EDA plots and return their paths."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = [
        ("target_distribution.png", _target_distribution(frame)),
        ("correlation_heatmap.png", _correlation_heatmap(frame)),
        ("key_relationships.png", _key_relationships(frame)),
    ]
    saved = []
    for filename, figure in figures:
        path = output / filename
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)
    return saved


def _target_distribution(frame: pd.DataFrame):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(frame[TARGET], bins=30, edgecolor="white")
    axis.set(title="House price distribution", xlabel="Price ($1000)", ylabel="Count")
    return figure


def _correlation_heatmap(frame: pd.DataFrame):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(frame.corr(numeric_only=True), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=90)
    axis.set_yticks(range(len(frame.columns)), frame.columns)
    figure.colorbar(image, ax=axis, label="Correlation")
    axis.set_title("Feature correlation matrix")
    return figure


def _key_relationships(frame: pd.DataFrame):
    import matplotlib.pyplot as plt

    features = [column for column in ("LSTAT", "RM") if column in frame]
    figure, axes = plt.subplots(1, len(features), figsize=(11, 4), squeeze=False)
    for axis, feature in zip(axes[0], features):
        axis.scatter(frame[feature], frame[TARGET], alpha=0.65, s=18)
        axis.set(xlabel=feature, ylabel="Price ($1000)", title=f"{feature} vs price")
    return figure