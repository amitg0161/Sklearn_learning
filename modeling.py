"""Reproducible regression training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelResult:
    name: str
    test_rmse: float
    test_r2: float
    cv_r2_mean: float
    cv_r2_std: float


def evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate baseline and regularized regressors on one split."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    cross_validator = KFold(n_splits=5, shuffle=True, random_state=random_state)
    models = {
        "linear_regression": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=10_000)),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        cv_scores = cross_val_score(model, features, target, cv=cross_validator, scoring="r2")
        result = ModelResult(
            name=name,
            test_rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
            test_r2=r2_score(y_test, predictions),
            cv_r2_mean=cv_scores.mean(),
            cv_r2_std=cv_scores.std(),
        )
        results.append(result.__dict__)
    return pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)