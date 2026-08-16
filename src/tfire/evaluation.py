"""Year-blocked cross-validation and the scoring functions every model is measured with."""

from __future__ import annotations

import logging
from collections.abc import Iterator
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
