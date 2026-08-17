"""SHAP attribution of the Trentino model"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.features.registry import Registry
from tfire.models.trentino import Estimator

logger = logging.getLogger(__name__)

# `season` is one categorical feature expanded into four indicators; attribution is summed back
# onto the registry name so the category totals stay comparable to the feature counts.
_DUMMY_SEPARATOR = "_"


@dataclass(frozen=True)
class Attribution:
    """Per-row SHAP values and the three aggregations they are read through."""

    values: npt.NDArray[np.float64]
    features: pd.DataFrame
    per_feature: pd.DataFrame
    per_category: pd.DataFrame
    per_temporal: pd.DataFrame


def registry_name(column: str, known: set[str]) -> str:
    """Map a design-matrix column back to the registry feature it came from."""
    if column in known:
        return column
    prefix = column.rsplit(_DUMMY_SEPARATOR, 1)[0]
    return prefix if prefix in known else column


def attribute(
    estimator: Estimator, features: pd.DataFrame, registry: Registry, dataset: pd.DataFrame
) -> Attribution:
    """Mean |SHAP| per feature, per registry category and per static/dynamic split."""
    import shap

    explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(features), dtype="float64")
    if values.shape != features.shape:
        raise ValueError(f"SHAP returned {values.shape} for a {features.shape} matrix")

    specs = {spec.name: spec for spec in registry.present(dataset)}
    origin = [registry_name(column, set(specs)) for column in features.columns]

    per_column = pd.DataFrame(
        {
            "column": features.columns,
            "feature": origin,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    per_column["category"] = [
        specs[name].category if name in specs else "unknown" for name in origin
    ]
    per_column["temporal"] = [
        specs[name].temporal if name in specs else "unknown" for name in origin
    ]

    summed = per_column.groupby(["feature", "category", "temporal"], as_index=False).agg(
        {"mean_abs_shap": "sum"}
    )
    per_feature = summed.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    total = float(per_feature["mean_abs_shap"].sum())
    per_feature["share"] = per_feature["mean_abs_shap"] / total

    per_category = _grouped(per_feature, "category")
    per_temporal = _grouped(per_feature, "temporal")
    logger.info(
        "SHAP over %d row(s): %.1f%% of the attribution is static, %.1f%% dynamic",
        len(features),
        100 * _share(per_temporal, "temporal", "static"),
        100 * _share(per_temporal, "temporal", "dynamic"),
    )
    return Attribution(values, features, per_feature, per_category, per_temporal)


def _grouped(per_feature: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = per_feature.groupby(column, as_index=False)[["mean_abs_shap", "share"]].sum()
    grouped["features"] = per_feature.groupby(column)[column].count().to_numpy()
    return grouped.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def _share(frame: pd.DataFrame, column: str, key: str) -> float:
    match = frame.loc[frame[column] == key, "share"]
    return float(match.iloc[0]) if len(match) else 0.0
