"""Random Forest and logistic regression, the points of comparison for the XGBoost model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tfire.config import Config

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


def _class_weight(weight: float) -> dict[int, float]:
    """Same ratio XGBoost gets through `scale_pos_weight`, so the models see one imbalance."""
    return {0: 1.0, 1: weight}


def random_forest(config: Config, weight: float) -> Pipeline:
    """Random forest with median imputation, the main benchmark for the XGBoost model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    settings = config.trentino.random_forest
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=settings.n_estimators,
                    min_samples_leaf=settings.min_samples_leaf,
                    class_weight=_class_weight(weight),
                    random_state=config.project.random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def logistic(config: Config, weight: float) -> Pipeline:
    """Also the FWI-only benchmark, which is this pipeline over a single column."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    settings = config.trentino.logistic
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=settings.c,
                    max_iter=settings.max_iter,
                    class_weight=_class_weight(weight),
                    random_state=config.project.random_seed,
                ),
            ),
        ]
    )
