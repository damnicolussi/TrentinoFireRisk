"""Blocked cross-validation in time and space, and the scores every model is measured with."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

Index = npt.NDArray[np.intp]


def year_blocks(years: pd.Series, n_folds: int) -> list[npt.NDArray[np.int64]]:
    """The distinct years split into `n_folds` contiguous groups of near-equal length."""
    distinct = np.sort(years.unique()).astype("int64")
    if len(distinct) < n_folds:
        raise ValueError(f"{len(distinct)} year(s) cannot fill {n_folds} folds")
    return [block for block in np.array_split(distinct, n_folds)]


def blocked_folds(years: pd.Series, n_folds: int) -> Iterator[tuple[Index, Index]]:
    """Positional train and validation indices, one pair per contiguous year block.

    Blocking is on the year rather than on the row because meteorology, drought state and
    vegetation are autocorrelated well past a single day: a random split would put a cell's
    neighbor in the training set on the same date and call the result generalization.
    """
    values = years.to_numpy("int64")
    positions = np.arange(len(values), dtype=np.intp)
    for block in year_blocks(years, n_folds):
        held = np.isin(values, block)
        yield positions[~held], positions[held]


def block_index(x: pd.Series, y: pd.Series, block_m: float) -> npt.NDArray[np.int64]:
    """One integer per row identifying the `block_m` square its coordinates fall in."""
    columns = np.floor(x.to_numpy("float64") / block_m).astype("int64")
    rows = np.floor(y.to_numpy("float64") / block_m).astype("int64")
    width = int(columns.max() - columns.min()) + 1
    flat: npt.NDArray[np.int64] = (rows - rows.min()) * width + (columns - columns.min())
    return flat


def spatial_folds(
    x: pd.Series, y: pd.Series, block_m: float, n_folds: int, seed: int
) -> Iterator[tuple[Index, Index]]:
    """Train and validation indices from square blocks of coordinates assigned to folds.

    Whole blocks move together, so a cell's neighbors are held out with it. Blocks are
    assigned at random rather than in reading order: contiguous assignment would make a fold
    one corner of the province and measure the corner rather than the model.
    """
    blocks = block_index(x, y, block_m)
    distinct = np.unique(blocks)
    if len(distinct) < n_folds:
        raise ValueError(f"{len(distinct)} block(s) cannot fill {n_folds} folds")

    shuffled = np.random.default_rng(seed).permutation(distinct)
    positions = np.arange(len(blocks), dtype=np.intp)
    for group in np.array_split(shuffled, n_folds):
        held = np.isin(blocks, group)
        yield positions[~held], positions[held]


def scores(labels: npt.NDArray[Any], probabilities: npt.NDArray[Any]) -> dict[str, float]:
    """AUPRC, AUROC and Brier, with the base rate the first two have to be read against."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    positive_rate = float(labels.mean())
    auprc = float(average_precision_score(labels, probabilities))
    return {
        "auprc": auprc,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "positive_rate": positive_rate,
        "lift": auprc / positive_rate,
    }


def bootstrap_scores(
    labels: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    n_resamples: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile confidence intervals for AUPRC, AUROC and lift, by resampling rows.

    Resampling is over rows rather than over positives alone, so the interval carries the
    uncertainty in the base rate as well: lift is defined against a rate the resample moves.
    A draw with no positive at all is discarded, since neither metric is defined on it.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(labels)
    drawn: dict[str, list[float]] = {"auprc": [], "auroc": [], "lift": []}
    for _ in range(n_resamples):
        rows = rng.integers(0, n, size=n)
        taken, scored = labels[rows], probabilities[rows]
        rate = float(taken.mean())
        if rate in (0.0, 1.0):
            continue
        auprc = float(average_precision_score(taken, scored))
        drawn["auprc"].append(auprc)
        drawn["auroc"].append(float(roc_auc_score(taken, scored)))
        drawn["lift"].append(auprc / rate)

    tail = 100 * (1 - level) / 2
    return {
        metric: {
            "lo": float(np.percentile(values, tail)),
            "hi": float(np.percentile(values, 100 - tail)),
            "resamples": len(values),
        }
        for metric, values in drawn.items()
    }


def precision_at_k(
    labels: npt.NDArray[Any], probabilities: npt.NDArray[Any], fractions: Sequence[float]
) -> list[dict[str, float]]:
    """Precision, recall and threshold over the top `fraction` of the ranking, per fraction.

    What a responder with capacity for the top q of the scored cells would actually catch.
    Ties at the cut are taken in the order `argsort` returns them, so a k landing inside a
    plateau of equal probabilities is arbitrary but reproducible.
    """
    order = np.argsort(-probabilities, kind="stable")
    ranked = labels[order]
    positives = int(labels.sum())

    rows = []
    for fraction in fractions:
        k = max(1, int(round(fraction * len(labels))))
        caught = int(ranked[:k].sum())
        rows.append(
            {
                "fraction": float(fraction),
                "k": k,
                "caught": caught,
                "precision": caught / k,
                "recall": caught / positives if positives else float("nan"),
                "threshold": float(probabilities[order[k - 1]]),
            }
        )
    return rows


def calibration_bins(
    labels: npt.NDArray[Any], probabilities: npt.NDArray[Any], n_bins: int
) -> list[dict[str, float]]:
    """Predicted against observed frequency over `n_bins` quantile bins of the probability.

    Quantile rather than equal-width: at a base rate this low nearly every prediction sits in
    the bottom tenth of the [0, 1] range and equal-width bins report nine empty rows.
    """
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    assigned = np.clip(np.searchsorted(edges, probabilities, side="left") - 1, 0, len(edges) - 2)

    rows = []
    for index in range(len(edges) - 1):
        members = assigned == index
        count = int(members.sum())
        if not count:
            continue
        rows.append(
            {
                "bin": index,
                "count": count,
                "predicted": float(probabilities[members].mean()),
                "observed": float(labels[members].mean()),
            }
        )
    return rows


def expected_calibration_error(bins: Sequence[dict[str, float]]) -> float:
    """Count-weighted mean gap between predicted and observed frequency."""
    total = sum(row["count"] for row in bins)
    if not total:
        raise ValueError("no populated calibration bins")
    return sum(row["count"] * abs(row["predicted"] - row["observed"]) for row in bins) / total
