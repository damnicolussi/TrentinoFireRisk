"""Has the fire regime changed, or the way fires are recorded? Period by period, read-only.

Writes `reports/nonstationarity.md` and its figures. Three parts: the cadastre and the training
table described by period, positives always against the same period's negatives; models fitted
on one period and scored on another, with the v2 hyperparameters; and a classifier asked to tell
pre-2000 from later ignitions. The recommended cut year and half-life are read off 1984-2014
alone, because 2015-2024 is the holdout the retrained variants are judged on.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config, load_config, setup_logging
from tfire.evaluation import blocked_folds, year_blocks
from tfire.features.registry import load_registry
from tfire.grid import load_grid
from tfire.models.explain import registry_name
from tfire.models.trentino import (
    SPECS,
    Estimator,
    design_matrix,
    final_params,
    positive_weight,
)
from tfire.report import table

logger = logging.getLogger("nonstationarity")

REPORT_FILENAME = "nonstationarity.md"

PERIODS = ((1984, 1999), (2000, 2014), (2015, 2024))
SPLIT_YEAR = 2000
LIGHTNING_CODE = 10

SIZE_CLASSES_HA = (0.0, 0.1, 1.0, 10.0, math.inf)
HOUR_BINS = (0, 3, 10, 14, 18, 22, 24)
BUILT_UP_SHARE = 0.3

DIAGONAL_FOLDS = 4
SIZE_MATCHED_REPEATS = 5
FIXED_RATIO = 10
RATIO_DRAWS = 200
TRANSFER_BLOCKS = 6
HALF_LIVES = (math.inf, 20.0, 10.0, 5.0, 3.0)
CUT_YEARS = (1994, 2000)
ADVERSARIAL_FOLDS = 5
TOP_FEATURES = 10

# products whose edition or sensor changes with the date, so they separate periods on their own
PRODUCT_CATEGORIES = ("landcover", "vegetation", "stacking")
PRODUCT_COLUMNS = ("pop_density",)

CORINE_L1 = {
    "CLC_L1_1": "artificial",
    "CLC_L1_2": "agricultural",
    "CLC_L1_3": "forest and semi-natural",
    "CLC_L1_4": "wetlands",
    "CLC_L1_5": "water",
}

CORINE_L2 = {
    "11": "urban fabric",
    "12": "industrial, commercial, transport",
    "13": "mine, dump, construction",
    "14": "artificial green",
    "21": "arable land",
    "22": "permanent crops",
    "23": "pastures",
    "24": "heterogeneous agricultural",
    "31": "forests",
    "32": "shrub and herbaceous",
    "33": "open, little vegetation",
    "41": "inland wetlands",
    "51": "inland waters",
}

# median for distances, mean for population, which is zero at most cells
CELL_SUMMARY = {
    "elevation_mean": "median",
    "dist_roads_mean": "median",
    "built_up_m": "median",
    "pop_density": "mean",
}

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def period_label(period: tuple[int, int]) -> str:
    return f"{period[0]}-{period[1]}"


def period_of(years: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=years.index, dtype="string")
    for period in PERIODS:
        labels[years.between(*period)] = period_label(period)
    return labels


def cadastre(config: Config) -> pd.DataFrame:
    """Every fire in the record, with its polygon's vertex count and mean segment length."""
    import shapely

    fires = pd.read_parquet(config.path(config.paths.fires_out))
    fires["year"] = fires["ignition_date"].dt.year
    fires = fires[fires["year"].between(PERIODS[0][0], PERIODS[-1][1])].copy()
    fires["period"] = period_of(fires["year"])

    polygons = pd.read_parquet(config.path(config.paths.fire_polygons_out))
    geometry = shapely.from_wkb(polygons["geometry"].to_numpy())
    shape = pd.DataFrame(
        {
            "fire_id": polygons["fire_id"].to_numpy(),
            "vertices": shapely.get_num_coordinates(geometry),
            "perimeter_m": shapely.length(geometry),
        }
    ).drop_duplicates("fire_id")
    fires = fires.merge(shape, on="fire_id", how="left", validate="m:1")
    fires["segment_m"] = fires["perimeter_m"] / fires["vertices"].clip(lower=1)
    return fires


def built_up_distance(config: Config) -> pd.DataFrame:
    """Distance from each cell to the nearest cell mostly artificial, per CORINE edition.

    There is no settlement layer in the data, so this stands in for distance to the built-up
    area. Cells outside the province count as not built up, which overstates the distance near
    the border.
    """
    from scipy.ndimage import distance_transform_edt

    spec, grid = load_grid(config)
    landcover = pd.read_parquet(
        config.path(config.paths.landcover_out), columns=["cell_id", "clc_edition", "CLC_L1_1"]
    )
    positions = grid.set_index("cell_id")[["y_index", "x_index"]]
    frames = []
    for edition, cells in landcover.groupby("clc_edition"):
        built = np.zeros((spec.n_rows, spec.n_cols), dtype=bool)
        where = positions.loc[cells.loc[cells["CLC_L1_1"] >= BUILT_UP_SHARE, "cell_id"]]
        built[where["y_index"].to_numpy(), where["x_index"].to_numpy()] = True
        distance = distance_transform_edt(~built) * spec.resolution_m
        at = positions.loc[cells["cell_id"]]
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cells["cell_id"].to_numpy(),
                    "clc_edition": edition,
                    "built_up_m": distance[at["y_index"].to_numpy(), at["x_index"].to_numpy()],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def per_year_counts(fires: pd.DataFrame) -> pd.DataFrame:
    """Fires per year in each size class, so a change in recording and in burning separate."""
    size = pd.cut(fires["area_ha"], SIZE_CLASSES_HA, right=False)
    counts = pd.crosstab(fires["period"], size)
    years = pd.Series({period_label(p): p[1] - p[0] + 1 for p in PERIODS})
    counts = counts.div(years, axis=0)
    counts["all"] = counts.sum(axis=1)
    return counts


def recording(fires: pd.DataFrame) -> pd.DataFrame:
    """The fields a change in how fires are logged would move."""
    minute = fires["start_datetime"].dt.minute
    day = fires["start_datetime"].dt.day
    grouped = fires.assign(
        minute_zero=minute == 0,
        minute_half=minute.isin((0, 30)),
        day_heaped=day.isin((1, 15)),
        tiny=fires["area_ha"] < SIZE_CLASSES_HA[1],
    ).groupby("period")
    return pd.DataFrame(
        {
            "fires": grouped.size(),
            "area_median": grouped["area_ha"].median(),
            "area_p90": grouped["area_ha"].quantile(0.9),
            "area_p99": grouped["area_ha"].quantile(0.99),
            "tiny": grouped["tiny"].mean(),
            "suspicious_hour": grouped["suspicious_default_time"].mean(),
            "minute_zero": grouped["minute_zero"].mean(),
            "minute_half": grouped["minute_half"].mean(),
            "day_heaped": grouped["day_heaped"].mean(),
            "vertices": grouped["vertices"].median(),
            "segment_m": grouped["segment_m"].median(),
            "centroid": grouped["ignition_point_method"].apply(lambda s: (s == "centroid").mean()),
        }
    )


def hours(fires: pd.DataFrame) -> pd.DataFrame:
    labels = [f"{a:02d}-{b - 1:02d}" for a, b in zip(HOUR_BINS[:-1], HOUR_BINS[1:], strict=True)]
    binned = pd.cut(fires["start_hour"].astype(float), HOUR_BINS, right=False, labels=labels)
    return pd.crosstab(fires["period"], binned, normalize="index")


def causes(fires: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(fires["cause"].fillna(-1).astype(int), fires["period"], normalize="columns")


def shares(table_: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share among positives, share among negatives and their ratio, per period."""
    rows = []
    for period, frame in table_.groupby("period"):
        positive = frame.loc[frame["is_fire"], column].value_counts(normalize=True)
        negative = frame.loc[~frame["is_fire"], column].value_counts(normalize=True)
        for value in positive.index.union(negative.index):
            rows.append(
                {
                    "period": period,
                    "value": value,
                    "positives": positive.get(value, 0.0),
                    "negatives": negative.get(value, 0.0),
                }
            )
    out = pd.DataFrame(rows)
    out["ratio"] = out["positives"] / out["negatives"].where(out["negatives"] > 0)
    return out


def cell_summary(table_: pd.DataFrame) -> pd.DataFrame:
    grouped = table_.groupby(["period", "is_fire"]).agg(CELL_SUMMARY)
    return pd.DataFrame(grouped.unstack("is_fire"))


def describe(config: Config, dataset: pd.DataFrame) -> dict[str, Any]:
    fires = cadastre(config)
    rows = dataset.copy()
    rows["period"] = period_of(rows["date"].dt.year)
    rows = rows.merge(built_up_distance(config), on=["cell_id", "clc_edition"], how="left")
    l1 = list(CORINE_L1)
    rows["corine_l1"] = rows[l1].idxmax(axis=1).map(CORINE_L1)
    l2 = [column for column in rows.columns if column.startswith("CLC_L2_")]
    rows["corine_l2"] = rows[l2].idxmax(axis=1).str.removeprefix("CLC_L2_").map(CORINE_L2)

    in_table = fires[fires["fire_id"].isin(dataset.loc[dataset["is_fire"], "fire_id"])]
    logger.info(
        "Cadastre: %d fires in the record, %d of them behind a training-table positive",
        len(fires),
        len(in_table),
    )
    return {
        "per_year": per_year_counts(fires),
        "recording": recording(fires),
        "hours": hours(fires),
        "causes": causes(fires),
        "season": shares(rows, "season"),
        "corine_l1": shares(rows, "corine_l1"),
        "corine_l2": shares(rows, "corine_l2"),
        "cells": cell_summary(rows),
        "positives": rows[rows["is_fire"]].groupby("period").size(),
        "negatives": rows[~rows["is_fire"]].groupby("period").size(),
        "fires": fires,
    }


def booster(config: Config, tuning: dict[str, Any], weight: float) -> Estimator:
    estimator = SPECS["xgboost"].build(config, weight)
    estimator.set_params(**final_params(tuning))
    return estimator


def fit_predict(
    config: Config,
    tuning: dict[str, Any],
    train: tuple[pd.DataFrame, npt.NDArray[np.int8]],
    test: pd.DataFrame,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    features, labels = train
    if sample_weight is None:
        weight = positive_weight(labels)
    else:
        positive = labels == 1
        weight = float(sample_weight[~positive].sum() / sample_weight[positive].sum())
    estimator = booster(config, tuning, weight)
    estimator.fit(features, labels, sample_weight=sample_weight, verbose=False)
    probabilities: FloatArray = estimator.predict_proba(test)[:, 1]
    return probabilities


def fixed_ratio(
    labels: npt.NDArray[np.int8], probabilities: FloatArray, rng: np.random.Generator
) -> dict[str, float]:
    """AUROC, and AUPRC at one positive per FIXED_RATIO negatives whatever the period's rate.

    The periods run from about 15% to 3.5% positives, so plain AUPRC would compare base rates.
    Whichever side is in excess is subsampled, without replacement, RATIO_DRAWS times.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    n_positive = min(len(positives), len(negatives) // FIXED_RATIO)
    n_negative = n_positive * FIXED_RATIO
    draws = np.empty(RATIO_DRAWS)
    for index in range(RATIO_DRAWS):
        kept = np.concatenate(
            [
                rng.choice(positives, n_positive, replace=False),
                rng.choice(negatives, n_negative, replace=False),
            ]
        )
        if int(labels[kept].sum()) * FIXED_RATIO != len(kept) - int(labels[kept].sum()):
            raise AssertionError("a fixed-ratio draw is off its ratio")
        draws[index] = average_precision_score(labels[kept], probabilities[kept])
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(draws.mean()),
        "auprc_low": float(np.quantile(draws, 0.025)),
        "auprc_high": float(np.quantile(draws, 0.975)),
    }


def subsample(
    labels: npt.NDArray[np.int8], n_positive: int, n_negative: int, rng: np.random.Generator
) -> IntArray:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    kept: IntArray = np.sort(
        np.concatenate(
            [
                rng.choice(positives, min(n_positive, len(positives)), replace=False),
                rng.choice(negatives, min(n_negative, len(negatives)), replace=False),
            ]
        )
    )
    return kept


def diagonal(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    sizes: tuple[int, int] | None,
    rng: np.random.Generator,
) -> FloatArray:
    """Out-of-fold probabilities from contiguous year blocks inside one period."""
    out = np.full(len(labels), np.nan)
    seen: list[npt.NDArray[np.intp]] = []
    for train_index, test_index in blocked_folds(years, DIAGONAL_FOLDS):
        if any(np.intersect1d(test_index, earlier).size for earlier in seen):
            raise AssertionError("diagonal folds overlap")
        seen.append(test_index)
        if sizes is not None:
            train_index = train_index[subsample(labels[train_index], *sizes, rng)]
        out[test_index] = fit_predict(
            config,
            tuning,
            (features.iloc[train_index], labels[train_index]),
            features.iloc[test_index],
        )
    return out


def matrix(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    sizes: tuple[int, int] | None,
    rng: np.random.Generator,
) -> dict[tuple[str, str], dict[str, float]]:
    """Trained on each period, scored on each period; the diagonal is cross-validated."""
    masks = {period_label(p): years.between(*p).to_numpy() for p in PERIODS}
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for source, inside in masks.items():
        index = np.flatnonzero(inside)
        own = diagonal(
            config,
            tuning,
            features.iloc[index],
            labels[index],
            years.iloc[index],
            sizes,
            rng,
        )
        cells[(source, source)] = fixed_ratio(labels[index], own, rng)

        train = index if sizes is None else index[subsample(labels[index], *sizes, rng)]
        others = [label for label in masks if label != source]
        held = np.flatnonzero(np.isin(np.arange(len(labels)), train, invert=True) & ~inside)
        scored = fit_predict(
            config,
            tuning,
            (features.iloc[train], labels[train]),
            features.iloc[held],
        )
        by_row = np.full(len(labels), np.nan)
        by_row[held] = scored
        for target in others:
            target_index = np.flatnonzero(masks[target])
            cells[(source, target)] = fixed_ratio(labels[target_index], by_row[target_index], rng)
        logger.info("Matrix row %s done%s", source, "" if sizes is None else " (size-matched)")
    return cells


def matched_sizes(labels: npt.NDArray[np.int8], years: pd.Series) -> tuple[int, int]:
    """Smallest positive and negative counts among every training set the matrix fits on."""
    positives, negatives = [], []
    for period in PERIODS:
        inside = years.between(*period).to_numpy()
        period_years = years[inside]
        period_labels = labels[inside]
        positives.append(int(period_labels.sum()))
        negatives.append(int((period_labels == 0).sum()))
        for train_index, _ in blocked_folds(period_years, DIAGONAL_FOLDS):
            positives.append(int(period_labels[train_index].sum()))
            negatives.append(int((period_labels[train_index] == 0).sum()))
    return min(positives), min(negatives)


def size_matched(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], dict[tuple[str, str], dict[str, float]]]:
    """The matrix again with every training set cut to the same size, repeated."""
    sizes = matched_sizes(labels, years)
    logger.info("Size-matched training sets: %d positives, %d negatives", *sizes)
    runs = [
        matrix(config, tuning, features, labels, years, sizes, rng)
        for _ in range(SIZE_MATCHED_REPEATS)
    ]
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key in runs[0]:
        values = np.array([run[key]["auroc"] for run in runs])
        precision = np.array([run[key]["auprc"] for run in runs])
        summary[key] = {
            "auroc": float(values.mean()),
            "auroc_low": float(values.min()),
            "auroc_high": float(values.max()),
            "auprc": float(precision.mean()),
        }
    return sizes, summary


def transfer_curve(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """Each 1984-2014 block trained alone and scored on every other, at equal training size.

    Returns the pairs and the slope of a fit with one intercept per target block, so that a
    target that is simply harder (later blocks are) does not read as drift.
    """
    from sklearn.metrics import roc_auc_score

    inside = (years < PERIODS[-1][0]).to_numpy()
    index = np.flatnonzero(inside)
    blocks = year_blocks(years.iloc[index], TRANSFER_BLOCKS)
    members = [index[np.isin(years.iloc[index].to_numpy(), block)] for block in blocks]
    n_positive = min(int(labels[m].sum()) for m in members)
    n_negative = min(int((labels[m] == 0).sum()) for m in members)
    rows = []
    for source, train in enumerate(members):
        kept = train[subsample(labels[train], n_positive, n_negative, rng)]
        rest = np.concatenate([m for k, m in enumerate(members) if k != source])
        scored = np.full(len(labels), np.nan)
        scored[rest] = fit_predict(
            config, tuning, (features.iloc[kept], labels[kept]), features.iloc[rest]
        )
        for target, test in enumerate(members):
            if target == source:
                continue
            gap = float(np.mean(blocks[target]) - np.mean(blocks[source]))
            rows.append(
                {
                    "source": f"{blocks[source][0]}-{blocks[source][-1]}",
                    "target": f"{blocks[target][0]}-{blocks[target][-1]}",
                    "gap_years": gap,
                    "auroc": float(roc_auc_score(labels[test], scored[test])),
                }
            )
    pairs = pd.DataFrame(rows)
    dummies = pd.get_dummies(pairs["target"], dtype=float)
    design = np.column_stack([dummies.to_numpy(), pairs["gap_years"].abs().to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, pairs["auroc"].to_numpy(), rcond=None)
    logger.info("Transfer curve: %d pairs, slope %.5f AUROC per year", len(pairs), coefficients[-1])
    return pairs, float(coefficients[-1])


def forward_scan(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Train on every year before a target block, weighted by recency or cut, score the block.

    The targets are the last two of v2's five training folds, so nothing after 2014 is seen.
    Weights decay as 0.5 ** (age / half-life), age counted from the last training year.
    """
    inside = (years < PERIODS[-1][0]).to_numpy()
    blocks = year_blocks(years[inside], config.trentino.cv_folds)[-2:]
    values = years.to_numpy("int64")
    rows = []
    for block in blocks:
        test = np.flatnonzero(np.isin(values, block))
        train = np.flatnonzero(values < block[0])
        last = int(values[train].max())
        variants: list[tuple[str, IntArray, FloatArray | None]] = []
        for half_life in HALF_LIVES:
            weights = None if math.isinf(half_life) else 0.5 ** ((last - values[train]) / half_life)
            name = "none" if math.isinf(half_life) else f"half-life {half_life:g} y"
            variants.append((name, train, weights))
        for cut in CUT_YEARS:
            variants.append((f"from {cut}", train[values[train] >= cut], None))
        for name, rows_, weights in variants:
            probabilities = fit_predict(
                config,
                tuning,
                (features.iloc[rows_], labels[rows_]),
                features.iloc[test],
                None if weights is None else np.asarray(weights, dtype="float64"),
            )
            rows.append(
                {
                    "target": f"{block[0]}-{block[-1]}",
                    "variant": name,
                    **fixed_ratio(labels[test], probabilities, rng),
                }
            )
        logger.info("Forward scan on %d-%d done", block[0], block[-1])
    return pd.DataFrame(rows)


def product_columns(columns: Sequence[str]) -> list[str]:
    registry = load_registry()
    known = {spec.name for spec in registry.features}
    category = {spec.name: spec.category for spec in registry.features}
    return [
        column
        for column in columns
        if category.get(registry_name(column, known)) in PRODUCT_CATEGORIES
        or registry_name(column, known) in PRODUCT_COLUMNS
    ]


def adversarial(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    dates: pd.Series,
) -> dict[str, Any]:
    """Cross-validated AUROC for before/after SPLIT_YEAR, and what the classifier leans on."""
    import xgboost
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    labels = (dates.dt.year >= SPLIT_YEAR).to_numpy("int8")
    out = np.full(len(labels), np.nan)
    for train, test in GroupKFold(ADVERSARIAL_FOLDS).split(features, labels, dates):
        out[test] = fit_predict(
            config, tuning, (features.iloc[train], labels[train]), features.iloc[test]
        )

    estimator = booster(config, tuning, positive_weight(labels))
    estimator.fit(features, labels, verbose=False)
    contributions = (
        cast(Any, estimator)
        .get_booster()
        .predict(xgboost.DMatrix(features), pred_contribs=True)[:, :-1]
    )
    known = {spec.name for spec in load_registry().features}
    importance = (
        pd.Series(np.abs(contributions).mean(axis=0), index=features.columns)
        .groupby(lambda column: registry_name(str(column), known))
        .sum()
        .sort_values(ascending=False)
    )
    return {
        "auroc": float(roc_auc_score(labels, out)),
        "rows": len(labels),
        "after": int(labels.sum()),
        "importance": importance,
        "medians": features.groupby(labels).median(),
    }


def adversarial_runs(
    config: Config,
    tuning: dict[str, Any],
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    dates: pd.Series,
    rng: np.random.Generator,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Positives against a same-size draw of negatives, with and without the dated products."""
    inside = (dates.dt.year < PERIODS[-1][0]).to_numpy()
    positives = np.flatnonzero(inside & (labels == 1))
    negatives = rng.choice(np.flatnonzero(inside & (labels == 0)), len(positives), replace=False)
    dropped = product_columns(list(features.columns))
    sets = {"all features": list(features.columns)}
    sets["no dated products"] = [c for c in features.columns if c not in set(dropped)]
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for rows_name, rows_ in (("positives", positives), ("negatives", np.sort(negatives))):
        for set_name, columns in sets.items():
            runs[(rows_name, set_name)] = adversarial(
                config,
                tuning,
                features.iloc[rows_][columns],
                dates.iloc[rows_].reset_index(drop=True),
            )
            logger.info(
                "Adversarial on %s, %s: AUROC %.3f",
                rows_name,
                set_name,
                runs[(rows_name, set_name)]["auroc"],
            )
    return runs


def figures(config: Config, fires: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    from matplotlib import pyplot as plt

    from tfire.figures import DASHES, SECONDARY_INK, SERIES, _label, _save, _style

    # six transfer blocks against five series slots
    colors = (*SERIES, SECONDARY_INK)
    dashes = (*DASHES, (0, (1, 1)))

    names = []
    with _style():
        figure, axes = plt.subplots(figsize=(7.0, 3.6))
        size = pd.cut(fires["area_ha"], SIZE_CLASSES_HA, right=False)
        counts = pd.crosstab(fires["year"], size)
        bounds = zip(SIZE_CLASSES_HA[:-1], SIZE_CLASSES_HA[1:], strict=True)
        for index, (column, (low, high)) in enumerate(zip(counts.columns, bounds, strict=True)):
            label = f"≥ {low:g} ha" if math.isinf(high) else f"{low:g}-{high:g} ha"
            axes.plot(
                counts.index,
                counts[column].clip(lower=0.5),
                color=SERIES[index],
                linestyle=DASHES[index],
                label=label,
            )
        axes.set_yscale("log")
        axes.legend(ncols=4, loc="upper right")
        _label(
            axes, "Fires per year by burned area", "year", "fires (log, a year with none at 0.5)"
        )
        names.append(_save(figure, config, "nonstationarity_sizes").name)

        figure, axes = plt.subplots(figsize=(7.0, 3.6))
        for index, period in enumerate(PERIODS):
            label = period_label(period)
            hour = fires.loc[fires["period"] == label, "start_hour"].astype(float)
            share = hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0)
            axes.plot(
                share.index,
                share.to_numpy(),
                color=SERIES[index],
                linestyle=DASHES[index],
                label=label,
            )
        axes.set_xticks(range(0, 24, 3))
        axes.legend()
        _label(axes, "Recorded hour of ignition", "hour", "share of the period's fires")
        names.append(_save(figure, config, "nonstationarity_hours").name)

        figure, axes = plt.subplots(figsize=(7.0, 3.6))
        for index, (target, frame) in enumerate(pairs.groupby("target")):
            frame = frame.sort_values("gap_years")
            axes.plot(
                frame["gap_years"],
                frame["auroc"],
                color=colors[index % len(colors)],
                linestyle=dashes[index % len(dashes)],
                marker="o",
                markersize=3,
                label=f"scored on {target}",
            )
        axes.axvline(0, color="#c3c2b7", linewidth=0.8)
        axes.legend(ncols=2, fontsize=7)
        _label(
            axes,
            "Transfer between 5-year blocks, equal training size",
            "target minus training block (years)",
            "AUROC",
        )
        names.append(_save(figure, config, "nonstationarity_transfer").name)
    return names


def ratio_text(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def matrix_rows(cells: dict[tuple[str, str], dict[str, float]], band: bool) -> list[list[str]]:
    labels = [period_label(p) for p in PERIODS]
    rows = []
    for source in labels:
        row = [source]
        for target in labels:
            cell = cells[(source, target)]
            text = f"{cell['auroc']:.3f}"
            if band:
                text += f" ({cell['auroc_low']:.3f}-{cell['auroc_high']:.3f})"
            text += f" / {cell['auprc']:.3f}"
            row.append(f"**{text}**" if source == target else text)
        rows.append(row)
    return rows


def recommend(
    full: dict[tuple[str, str], dict[str, float]],
    matched: dict[tuple[str, str], dict[str, float]],
    scan: pd.DataFrame,
) -> dict[str, Any]:
    """The cut year and half-life worth trying, read off 1984-2014 only."""
    early, late = (period_label(p) for p in PERIODS[:2])
    own, carried = matched[(late, late)], matched[(early, late)]
    cut = own["auroc_low"] > carried["auroc_high"]
    mean = scan.groupby("variant")[["auroc", "auprc"]].mean()
    baseline = float(mean["auroc"].loc["none"])
    half = mean[mean.index.str.startswith("half-life")]
    best = str(half["auroc"].idxmax())
    return {
        "own": own,
        "carried": carried,
        "cut_2000": bool(cut),
        "full_gap": full[(late, late)]["auroc"] - full[(early, late)]["auroc"],
        "mean": mean,
        "best_half_life": best,
        "cut_loss": {cut: baseline - float(mean["auroc"].loc[f"from {cut}"]) for cut in CUT_YEARS},
        "best_gain": float(half["auroc"].max()) - baseline,
    }


def render(results: dict[str, Any]) -> str:
    described = results["describe"]
    labels = [period_label(p) for p in PERIODS]
    lines = [
        "# Non-stationarity of the fire cadastre",
        "",
        "Generated by `scripts/nonstationarity.py`. Periods: "
        + ", ".join(labels)
        + ". Positives are always read against the same period's negatives, which are a uniform "
        "draw over the record and so describe the background each period scored against.",
        "",
        "**Holdout rule.** 2015-2024 is the holdout retrained variants are judged on. It is "
        "described here and appears in the matrix, but the recommendation at the end uses "
        "1984-2014 only.",
        "",
        "## 1. The cadastre by period",
        "",
        "### Fires per year by burned area",
        "",
    ]
    per_year = described["per_year"]
    size_headers = [str(column) for column in per_year.columns]
    lines += table(
        ["period", *size_headers],
        [[p, *(f"{v:.1f}" for v in per_year.loc[p])] for p in labels],
    )
    lines += [
        "",
        "Intervals are [lower, upper) in hectares. A better-recorded cadastre would show the "
        "small classes growing in absolute count while the large ones hold; a changed regime "
        "shows every class falling.",
        "",
        f"![sizes](figures/{results['figures'][0]})",
        "",
        "### Recording fields",
        "",
    ]
    recorded = described["recording"]
    lines += table(
        [
            "period",
            "fires",
            "area median (ha)",
            "p90",
            "p99",
            "< 0.1 ha",
            "hour 00-02",
            "minute :00",
            "minute :00/:30",
            "day 1st/15th",
            "vertices (median)",
            "segment (m, median)",
        ],
        [
            [
                p,
                int(r["fires"]),
                f"{r['area_median']:.3f}",
                f"{r['area_p90']:.2f}",
                f"{r['area_p99']:.1f}",
                pct(r["tiny"]),
                pct(r["suspicious_hour"]),
                pct(r["minute_zero"]),
                pct(r["minute_half"]),
                pct(r["day_heaped"]),
                f"{r['vertices']:.0f}",
                f"{r['segment_m']:.1f}",
            ]
            for p, r in recorded.loc[labels].iterrows()
        ],
    )
    lines += [
        "",
        "Hour 00-02 is `suspicious_default_time`, the band where placeholder times pile up. Day "
        "heaping would be about 6.6% by chance. Segment is the polygon perimeter over its "
        "vertex count, shorter for field GPS tracks than for drawn outlines.",
        "",
        f"![hours](figures/{results['figures'][1]})",
        "",
        "### Hour of ignition",
        "",
    ]
    hour = described["hours"]
    lines += table(
        ["period", *hour.columns.astype(str)],
        [[p, *(pct(v) for v in hour.loc[p])] for p in labels],
    )
    lines += ["", "### Cause codes", ""]
    cause = described["causes"]
    lines += table(
        ["code", *labels],
        [
            [
                "missing"
                if code == -1
                else f"{code}" + (" (lightning)" if code == LIGHTNING_CODE else ""),
                *(pct(cause.loc[code, p]) for p in labels),
            ]
            for code in cause.index
        ],
    )
    lines += [
        "",
        f"Only code {LIGHTNING_CODE} has a known meaning; the cadastre ships no legend for the "
        "others.",
        "",
        "## 2. Where and when, against the background",
        "",
        "Share among positives over share among negatives of the same period. Above 1, ignitions "
        "are over-represented there.",
        "",
    ]
    for key, title in (
        ("season", "Season"),
        ("corine_l1", "CORINE level 1, dominant class of the cell"),
        ("corine_l2", "CORINE level 2, dominant class, classes with at least 2% of positives"),
    ):
        frame = described[key]
        pivot = frame.pivot(index="value", columns="period", values=["positives", "ratio"])
        pivot["positives"] = pivot["positives"].fillna(0.0)
        if key == "corine_l2":
            pivot = pivot[pivot["positives"].max(axis=1) >= 0.02]
        lines += [f"### {title}", ""]
        lines += table(
            ["", *(f"{p} share" for p in labels), *(f"{p} ratio" for p in labels)],
            [
                [
                    value,
                    *(pct(pivot.loc[value, ("positives", p)]) for p in labels),
                    *(ratio_text(pivot.loc[value, ("ratio", p)]) for p in labels),
                ]
                for value in pivot.index
            ],
        )
        lines += [""]
    lines += [
        "### The cells fires start in, positives against negatives",
        "",
        "Distance to the built-up area is to the nearest cell at least "
        f"{BUILT_UP_SHARE:.0%} CORINE artificial surfaces, edition matched to the date; there is "
        "no settlement layer in the data. Road distance is from today's OSM network for every "
        "year, so a change here is a change in where fires are, not in the roads. Population is "
        "the mean of the density feature, whose median is zero.",
        "",
    ]
    cells = described["cells"]
    lines += table(
        ["period", *(f"{c}, {how} (pos / neg)" for c, how in CELL_SUMMARY.items())],
        [
            [
                p,
                *(
                    f"{cells.loc[p, (c, True)]:.1f} / {cells.loc[p, (c, False)]:.1f}"
                    for c in CELL_SUMMARY
                ),
            ]
            for p in labels
        ],
    )

    sizes = results["sizes"]
    lines += [
        "",
        "## 3. Trained on one period, scored on another",
        "",
        "v2 hyperparameters, "
        f"{results['rounds']} rounds, no early stopping. Rows are the training period, columns "
        "the scored one. Each cell is AUROC / AUPRC at a fixed 1:"
        f"{FIXED_RATIO} ratio (mean of {RATIO_DRAWS} draws), so neither depends on the period's "
        f"base rate. The diagonal is cross-validated on {DIAGONAL_FOLDS} contiguous year blocks "
        "inside the period.",
        "",
        "### Full training sets",
        "",
    ]
    lines += table(["trained on", *labels], matrix_rows(results["matrix"], band=False))
    lines += [
        "",
        f"### Equal training size ({sizes[0]} positives, {sizes[1]} negatives, "
        f"{SIZE_MATCHED_REPEATS} repeats, AUROC range in brackets)",
        "",
        "The smallest training set any cell above uses. It separates less data from different "
        "data.",
        "",
    ]
    lines += table(["trained on", *labels], matrix_rows(results["matched"], band=True))

    slope = results["transfer"][1]
    lines += [
        "",
        "### Transfer between 5-year blocks, 1984-2014",
        "",
        f"Each block trained alone at equal size and scored on every other. A fit with one "
        f"intercept per scored block gives **{slope * 10:+.4f} AUROC per decade** of distance "
        "between training and scored years.",
        "",
        f"![transfer](figures/{results['figures'][2]})",
        "",
        "### Recency weights and cuts, forward in time",
        "",
        "Trained on every year before the scored block, which is one of the last two v2 training "
        "folds. Weights decay as 0.5^(age / half-life) from the last training year.",
        "",
    ]
    scan = results["scan"]
    targets = list(dict.fromkeys(scan["target"]))
    variants = list(dict.fromkeys(scan["variant"]))
    lines += table(
        ["variant", *(f"{t} AUROC / AUPRC" for t in targets), "mean AUROC"],
        [
            [
                variant,
                *(
                    "{:.3f} / {:.3f}".format(
                        *scan.loc[
                            (scan["variant"] == variant) & (scan["target"] == target),
                            ["auroc", "auprc"],
                        ].iloc[0]
                    )
                    for target in targets
                ),
                f"{scan.loc[scan['variant'] == variant, 'auroc'].mean():.3f}",
            ]
            for variant in variants
        ],
    )

    lines += [
        "",
        "## 4. Telling pre-2000 ignitions from later ones",
        "",
        f"Cross-validated AUROC of a classifier for year ≥ {SPLIT_YEAR}, 1984-2014, folds grouped "
        "by date. Run on the positives and, as the control for background drift (climate, "
        "product editions), on as many negatives. 'No dated products' drops CORINE, WorldPop, "
        "Landsat and the Mesogeos score, whose edition or sensor follows the date.",
        "",
    ]
    runs = results["adversarial"]
    early_label, late_label = (period_label(p) for p in PERIODS[:2])
    set_names = ["all features", "no dated products"]
    lines += table(
        ["rows", *set_names],
        [
            [rows_, *(f"{runs[(rows_, s)]['auroc']:.3f}" for s in set_names)]
            for rows_ in ("positives", "negatives")
        ],
    )
    lines += [
        "",
        f"Top {TOP_FEATURES} features by mean |SHAP|, no dated products:",
        "",
    ]
    positive_run = runs[("positives", "no dated products")]
    negative_run = runs[("negatives", "no dated products")]
    positive, negative = positive_run["importance"], negative_run["importance"]
    top = positive.head(TOP_FEATURES)

    def shift(run: dict[str, Any], name: str) -> str:
        medians = run["medians"]
        if name not in medians.columns:
            return "n/a"
        return f"{medians.loc[0, name]:.1f} → {medians.loc[1, name]:.1f}"

    lines += table(
        [
            "feature",
            "|SHAP| positives",
            "|SHAP| negatives",
            "ratio",
            "positives, median before → after",
            "negatives, median before → after",
        ],
        [
            [
                name,
                f"{value:.3f}",
                f"{negative.get(name, 0.0):.3f}",
                f"{value / negative.get(name, math.nan):.1f}",
                shift(positive_run, name),
                shift(negative_run, name),
            ]
            for name, value in top.items()
        ],
    )
    season = described["season"].pivot(index="value", columns="period", values="positives")
    lines += [
        "",
        "The weather that separates the positives also encodes the season, and the seasonal mix "
        f"is what moved: summer went from {pct(season.loc['summer', early_label])} to "
        f"{pct(season.loc['summer', late_label])} of ignitions. The wetter antecedent week "
        "after 2000 mostly reflects summer fires, which follow summer rain, replacing dry-winter "
        "ones.",
    ]

    choice = results["recommendation"]
    early, late = early_label, late_label
    per_year = described["per_year"]
    fell = per_year.loc[late] / per_year.loc[early]
    lines += [
        "",
        "## 5. What a retraining should try",
        "",
        "- **Regime or recording.** Between "
        f"{early} and {late}, fires per year fell in every size class: "
        + ", ".join(
            f"{column} to {pct(value)}" for column, value in fell.items() if column != "all"
        )
        + ". The smallest class fell least, which better recording could explain; the fall in "
        "every class it cannot. Burning changed. "
        "Recording also changed, separately: finer polygons and fewer round minutes, and in "
        "2015-2024 a placeholder hour on one fire in five.",
        f"- **Cut year.** At equal training size, {late} scored by its own years gets AUROC "
        f"{choice['own']['auroc']:.3f} ({choice['own']['auroc_low']:.3f}-"
        f"{choice['own']['auroc_high']:.3f}), by {early} {choice['carried']['auroc']:.3f} "
        f"({choice['carried']['auroc_low']:.3f}-{choice['carried']['auroc_high']:.3f}). "
        + (
            "The ranges do not overlap: training from 2000 is worth a variant, with 1994 as the "
            "intermediate control."
            if choice["cut_2000"]
            else "The ranges overlap: the older years cost nothing measurable at equal size, so "
            "a cut at 2000 is a control, not the expected winner."
        ),
        f"- **Half-life.** Forward in time, the best recency weighting is "
        f"**{choice['best_half_life']}**, {choice['best_gain']:+.4f} AUROC over no weights "
        "(mean of the two scored blocks), while training from 1994 or 2000 loses "
        f"{choice['cut_loss'][1994]:.3f} and {choice['cut_loss'][2000]:.3f}. "
        "Pooled AUROC barely moves with recency, so a retraining should expect little there and "
        "judge the variants where the change is, on summer and on the event axis. Half-lives of "
        "5, 10 and 20 years bracket the result.",
        "- **Not fixable by weights.** Recording changes in the table above (placeholder hours, "
        "tiny fires, polygon detail) move what a positive is, not how much it should count. "
        "They are data notes, not a weighting problem.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)
    rng = np.random.default_rng(config.project.random_seed)

    dataset = pd.read_parquet(config.path(config.paths.dataset_out))
    metrics_path = config.path(config.paths.trentino_model_dir) / "v2" / "metrics.json"
    tuning = json.loads(metrics_path.read_text())["tuning"]

    described = describe(config, dataset)
    counts = described["positives"]
    if int(counts.sum()) != int(dataset["is_fire"].sum()):
        raise AssertionError("period positives do not add up to the training table")

    features, labels, years = design_matrix(dataset, load_registry())
    full = matrix(config, tuning, features, labels, years, None, rng)
    sizes, matched = size_matched(config, tuning, features, labels, years, rng)
    pairs, slope = transfer_curve(config, tuning, features, labels, years, rng)
    scan = forward_scan(config, tuning, features, labels, years, rng)
    runs = adversarial_runs(config, tuning, features, labels, dataset["date"], rng)

    results: dict[str, Any] = {
        "describe": described,
        "rounds": tuning["rounds"],
        "matrix": full,
        "sizes": sizes,
        "matched": matched,
        "transfer": (pairs, slope),
        "scan": scan,
        "adversarial": runs,
        "recommendation": recommend(full, matched, scan),
        "figures": figures(config, described["fires"], pairs),
    }
    out = config.path(config.paths.report_dir) / REPORT_FILENAME
    out.write_text(render(results), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
