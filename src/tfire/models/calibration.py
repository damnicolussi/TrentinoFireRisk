"""Isotonic recalibration of the probabilities, and the offset back to the real base rate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.evaluation import calibration_bins, expected_calibration_error, scores
from tfire.sampling import negative_pool

logger = logging.getLogger(__name__)

CALIBRATOR_FILENAME = "calibrator.json"

_EPSILON = 1e-9


@dataclass(frozen=True)
class Calibrator:
    """Isotonic knots plus the case-control offset, everything inference needs to score."""

    thresholds: list[float]
    values: list[float]
    log_offset: float
    sampling_rate: float
    counts: dict[str, int]

    def to_sample_rate(self, probabilities: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Calibrated against the case-control sample the model was trained on."""
        mapped: npt.NDArray[np.float64] = np.interp(probabilities, self.thresholds, self.values)
        return mapped

    def to_population_rate(self, probabilities: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Calibrated against the real rate of a cell-day burning."""
        return _inverse_logit(_logit(self.to_sample_rate(probabilities)) + self.log_offset)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Calibrator:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _logit(p: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    clipped = np.clip(np.asarray(p, dtype="float64"), _EPSILON, 1 - _EPSILON)
    odds: npt.NDArray[np.float64] = np.log(clipped / (1 - clipped))
    return odds


def _inverse_logit(z: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    return np.asarray(1 / (1 + np.exp(-z)), dtype="float64")


def sampling_rate(config: Config, n_sampled_negatives: int) -> tuple[float, dict[str, int]]:
    """The share of the population's non-fire cell-days that made it into the training table.

    Positives are taken whole and negatives are subsampled, so the sample's base rate is an
    artifact of the draw. This is the number that undoes it.
    """
    grid = pd.read_parquet(config.path(config.paths.grid_out))
    exclusions = pd.read_parquet(config.path(config.paths.exclusions_out))
    pool, days, blocked = negative_pool(grid, exclusions, config)

    population = len(pool) * len(days) - len(blocked)
    counts = {
        "cells": len(pool),
        "days": len(days),
        "excluded_cell_days": len(blocked),
        "population_negatives": population,
        "sampled_negatives": n_sampled_negatives,
    }
    return n_sampled_negatives / population, counts


def isotonic_knots(
    labels: npt.NDArray[Any], probabilities: npt.NDArray[Any]
) -> tuple[list[float], list[float]]:
    """The step function mapping a predicted probability to the frequency observed beside it."""
    from sklearn.isotonic import IsotonicRegression

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(probabilities, labels)
    return (
        [float(value) for value in isotonic.X_thresholds_],
        [float(value) for value in isotonic.y_thresholds_],
    )


def fit(
    labels: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    config: Config,
    n_negatives: int,
) -> Calibrator:
    """Isotonic regression on the out-of-fold predictions, plus the case-control offset.

    Fitted out of fold rather than on the training predictions, which the model has already
    driven to near-separation. `n_negatives` counts the whole table, not the fitting span.
    """
    thresholds, values = isotonic_knots(labels, probabilities)
    rate, counts = sampling_rate(config, n_negatives)
    logger.info(
        "Negatives sampled at 1 in %.0f of %d population cell-days, log-odds offset %.3f",
        1 / rate,
        counts["population_negatives"],
        float(np.log(rate)),
    )
    return Calibrator(
        thresholds=thresholds,
        values=values,
        log_offset=float(np.log(rate)),
        sampling_rate=rate,
        counts=counts,
    )


def report(
    labels: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    calibrator: Calibrator,
    n_bins: int,
) -> dict[str, Any]:
    """Reliability of the raw and the recalibrated probabilities on the same rows."""
    calibrated = calibrator.to_sample_rate(probabilities)

    blocks = {}
    for name, values in (("raw", probabilities), ("isotonic", calibrated)):
        bins = calibration_bins(labels, values, n_bins)
        blocks[name] = {
            "bins": bins,
            "ece": expected_calibration_error(bins),
            "brier": scores(labels, values)["brier"],
            "mean_predicted": float(np.mean(values)),
        }

    population = calibrator.to_population_rate(probabilities)
    blocks["population"] = {
        "mean_predicted": float(np.mean(population)),
        "max_predicted": float(np.max(population)),
        "observed_sample_rate": float(labels.mean()),
    }
    logger.info(
        "Calibration on the holdout: ECE %.4f raw, %.4f isotonic | Brier %.4f raw, %.4f isotonic",
        blocks["raw"]["ece"],
        blocks["isotonic"]["ece"],
        blocks["raw"]["brier"],
        blocks["isotonic"]["brier"],
    )
    return blocks
