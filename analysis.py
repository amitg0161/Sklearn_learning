"""Tabular summaries used by the EDA and modeling workflows."""

from __future__ import annotations

import pandas as pd

from data import TARGET


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-column quality and descriptive statistics."""
    summary = frame.describe().T
    summary["missing"] = frame.isna().sum()
    summary["dtype"] = frame.dtypes.astype(str)
    return summary


def target_correlations(frame: pd.DataFrame) -> pd.Series:
    """Return feature correlations with the target, ordered by magnitude."""
    correlations = frame.corr(numeric_only=True)[TARGET].drop(TARGET)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)