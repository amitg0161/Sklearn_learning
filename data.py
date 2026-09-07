"""Dataset loading and validation for the Boston housing example."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml


TARGET = "PRICE"


def load_housing_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Boston housing data from OpenML."""
    dataset = fetch_openml(name="boston", version=1, as_frame=True)
    features = dataset.data.apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(dataset.target, errors="raise").rename(TARGET)

    if features.empty or len(features) != len(target):
        raise ValueError("The housing dataset has inconsistent feature and target rows.")
    return features, target


def build_frame(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Combine features and target into one analysis-friendly frame."""
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows.")
    return features.assign(**{TARGET: target.to_numpy()})