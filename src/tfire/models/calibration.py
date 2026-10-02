"""Isotonic recalibration of the probabilities, and the offset back to the real base rate."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
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

# Krichevsky-Trofimov: half a count on each side of every isotonic block
_DAMPING = 0.5


@dataclass(frozen=True)
class Calibrator:
    """Isotonic knots plus the case-control offset, everything inference needs to score."""

    thresholds: list[float]
    values: list[float]
    log_offset: float
    sampling_rate: float
    counts: dict[str, int]
    window_offset: float = 0.0
    reference_years: list[int] | None = None
    window: dict[str, Any] = field(default_factory=dict)

    def to_sample_rate(self, probabilities: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Calibrated against the case-control sample the model was trained on."""
        mapped: npt.NDArray[np.float64] = np.interp(probabilities, self.thresholds, self.values)
        return mapped

    def to_population_rate(self, probabilities: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Calibrated against the real rate of a cell-day burning."""
        return _inverse_logit(self.record_logit(probabilities) + self.window_offset)

    def record_logit(self, probabilities: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Log-odds on the average rate of the years the model was fitted on."""
        return _logit(self.to_sample_rate(probabilities)) + self.log_offset

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

    Matched negatives are drawn from the cells that burned rather than from the whole pool, so
    they are not a uniform sample of anything and cannot enter the offset. Only the uniform
    part of the draw does; what the matched part does to the shape is left to the isotonic fit.
    """
    grid = pd.read_parquet(config.path(config.paths.grid_out))
    exclusions = pd.read_parquet(config.path(config.paths.exclusions_out))
    pool, days, blocked = negative_pool(grid, exclusions, config)

    population = len(pool) * len(days) - len(blocked)
    matched = int(round(config.sampling.hard_negative_fraction * n_sampled_negatives))
    uniform = n_sampled_negatives - matched

    counts = {
        "cells": len(pool),
        "days": len(days),
        "excluded_cell_days": len(blocked),
        "population_negatives": population,
        "sampled_negatives": n_sampled_negatives,
        "matched_negatives": matched,
        "uniform_negatives": uniform,
    }
    return uniform / population, counts


def isotonic_knots(
    labels: npt.NDArray[Any], probabilities: npt.NDArray[Any]
) -> tuple[list[float], list[float]]:
    """The step function mapping a predicted probability to the frequency observed beside it.

    Each pooled block is reported as `(positives + 0.5) / (size + 1)` rather than its raw
    frequency, so a block of only positives lands below 1. An exact 1.0 survives the case-control
    offset: the logit clips at `1 - _EPSILON`, which is +20.7, and an offset of about -9 leaves
    it at 0.9999. The map then claims a cell will certainly burn, on the strength of a leaf
    holding a handful of rows.

    Damping a block on its own count is not monotone (a one-row block of a single negative gives
    0.25, a hundred-row block at frequency 0.1 gives 0.104), so a running maximum restores the
    ordering isotonic regression exists to produce. It cannot reintroduce a 1.0: every damped
    value is strictly below it.
    """
    from sklearn.isotonic import IsotonicRegression

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    fitted = isotonic.fit_transform(probabilities, labels)

    blocks, index = np.unique(fitted, return_inverse=True)
    sizes = np.bincount(index, minlength=len(blocks))
    hits = np.bincount(index, weights=np.asarray(labels, dtype="float64"), minlength=len(blocks))
    damped = np.maximum.accumulate((hits + _DAMPING) / (sizes + 2 * _DAMPING))

    return (
        [float(value) for value in isotonic.X_thresholds_],
        [float(value) for value in np.interp(isotonic.y_thresholds_, blocks, damped)],
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


@dataclass(frozen=True)
class ObservedRate:
    """Ignitions per population cell-day over a span of years, with its exact 95% interval."""

    years: list[int]
    ignitions: int
    cell_days: int
    rate: float
    low: float
    high: float


def observed_rate(config: Config, first: int, last: int) -> ObservedRate:
    """The rate a cell-day burned in `first`-`last`, on the population the negatives came from."""
    grid = pd.read_parquet(config.path(config.paths.grid_out))
    exclusions = pd.read_parquet(config.path(config.paths.exclusions_out))
    samples = pd.read_parquet(config.path(config.paths.samples_out), columns=["date", "is_fire"])
    pool, days, blocked = negative_pool(grid, exclusions, config)
    return count_rate(
        len(pool), days, blocked, samples.loc[samples["is_fire"], "date"], first, last
    )


def count_rate(
    n_cells: int,
    days: pd.DatetimeIndex,
    blocked: npt.NDArray[np.int64],
    ignitions: pd.Series,
    first: int,
    last: int,
) -> ObservedRate:
    """Ignitions over the unexcluded cell-days of `first`-`last`."""
    from scipy.stats import chi2

    inside = (days.year >= first) & (days.year <= last)
    if not inside.any():
        raise ValueError(f"{first}-{last} is outside the record {days[0]:%Y}-{days[-1]:%Y}")
    offsets = np.flatnonzero(inside)
    day_of = blocked % len(days)
    excluded = int(((day_of >= offsets[0]) & (day_of <= offsets[-1])).sum())
    cell_days = n_cells * len(offsets) - excluded

    year = pd.DatetimeIndex(ignitions).year
    count = int(((year >= first) & (year <= last)).sum())
    low = chi2.ppf(0.025, 2 * count) / 2 if count else 0.0
    high = chi2.ppf(0.975, 2 * count + 2) / 2
    return ObservedRate(
        years=[first, last],
        ignitions=count,
        cell_days=cell_days,
        rate=count / cell_days,
        low=float(low) / cell_days,
        high=float(high) / cell_days,
    )


def sampled_days(first: int, last: int, stride: int) -> list[date]:
    start, end = date(first, 1, 1), date(last, 12, 31)
    return [start + timedelta(days=offset) for offset in range(0, (end - start).days + 1, stride)]


def mean_rate(logits: npt.NDArray[np.float64], shift: float) -> float:
    return float(_inverse_logit(logits + shift).mean())


def empirical_shift(logits: npt.NDArray[np.float64], target: float) -> float:
    """The log-odds shift that makes the mean probability over `logits` equal `target`."""
    from scipy.optimize import brentq

    def gap(shift: float) -> float:
        return mean_rate(logits, shift) - target

    return float(brentq(gap, -20.0, 20.0, xtol=1e-6))


def window_shift(
    logits: npt.NDArray[np.float64], window: ObservedRate, fitted: ObservedRate
) -> dict[str, Any]:
    """Both estimates of the shift to the window's rate, and which one is used."""
    prior = float(np.log(window.rate / fitted.rate))
    empirical = empirical_shift(logits, window.rate)
    prior_mean = mean_rate(logits, prior)
    inside = window.low <= prior_mean <= window.high

    return {
        "window": asdict(window),
        "fitted_on": asdict(fitted),
        "prior_shift": prior,
        "empirical_shift": empirical,
        "unshifted_mean": mean_rate(logits, 0.0),
        "prior_mean": prior_mean,
        "chosen": "prior" if inside else "empirical",
        "shift": prior if inside else empirical,
        "reason": "the prior shift's mean falls inside the window's 95% interval"
        if inside
        else f"the prior shift gives a mean of {prior_mean:.3e}, outside the window's "
        f"interval {window.low:.3e}-{window.high:.3e}",
    }


def _logits_by_day(
    config: Config, days: list[date], holdout: bool
) -> dict[date, npt.NDArray[np.float32]]:
    """Record-rate log-odds over the whole grid for each day."""
    from tfire.inference import GridScorer

    scorer = GridScorer(config, days, holdout=holdout)
    logger.info("Scoring %d whole day(s) for the calibration window", len(days))
    logits = {}
    for index, day in enumerate(days):
        logits[day] = scorer.calibrator.record_logit(scorer.day(day).raw).astype("float32")
        if index and index % 100 == 0:
            logger.info("  %d/%d days", index, len(days))
    return logits


def _stack(
    logits: dict[date, npt.NDArray[np.float32]], first: int, last: int
) -> npt.NDArray[np.float64]:
    picked = [values for day, values in logits.items() if first <= day.year <= last]
    return np.concatenate(picked).astype("float64")


def reference_window(config: Config) -> tuple[int, int]:
    last = config.date_range.end.year
    return last - config.calibration.window_years + 1, last


def apply_window(config: Config) -> Calibrator:
    """Shift the stored calibrator onto the rate of the reference window, and write it back."""
    from tfire.inference import model_directory

    path = model_directory(config) / CALIBRATOR_FILENAME
    stored = Calibrator.read(path)

    first, last = reference_window(config)
    window = observed_rate(config, first, last)
    fitted = observed_rate(config, config.date_range.start.year, config.date_range.end.year)
    days = sampled_days(first, last, config.calibration.empirical_stride_days)
    logits = _stack(_logits_by_day(config, days, holdout=False), first, last)

    decided = window_shift(logits, window, fitted)
    decided["days"] = len(days)
    calibrator = Calibrator(
        **{
            **stored.__dict__,
            "window_offset": decided["shift"],
            "reference_years": [first, last],
            "window": decided,
        }
    )
    calibrator.write(path)
    logger.info(
        "Calibration window %d-%d: %d ignition(s), rate %.3e | prior shift %.3f, empirical %.3f, "
        "using the %s one",
        first,
        last,
        window.ignitions,
        window.rate,
        decided["prior_shift"],
        decided["empirical_shift"],
        decided["chosen"],
    )
    return calibrator


def validate_window(config: Config) -> list[dict[str, Any]]:
    """Each validation window's shift, fitted before the holdout years and scored on them."""
    stride = config.calibration.empirical_stride_days
    end_fit = config.trentino.test_years_start - 1
    test_first, test_last = config.trentino.test_years_start, config.date_range.end.year
    widest = max(config.calibration.validation_windows)

    days = sampled_days(end_fit - widest + 1, test_last, stride)
    logits = _logits_by_day(config, days, holdout=True)

    fitted = observed_rate(config, config.date_range.start.year, end_fit)
    after = observed_rate(config, test_first, test_last)
    target = _stack(logits, test_first, test_last)

    rows = []
    for length in sorted(config.calibration.validation_windows):
        window = observed_rate(config, end_fit - length + 1, end_fit)
        decided = window_shift(_stack(logits, end_fit - length + 1, end_fit), window, fitted)
        predicted = mean_rate(target, decided["shift"])
        rows.append(
            {
                "window_years": length,
                "window": window.years,
                "window_rate": window.rate,
                "prior_shift": decided["prior_shift"],
                "empirical_shift": decided["empirical_shift"],
                "chosen": decided["chosen"],
                "predicted_after": predicted,
                "predicted_after_prior": mean_rate(target, decided["prior_shift"]),
                "predicted_after_empirical": mean_rate(target, decided["empirical_shift"]),
                "unshifted_after": mean_rate(target, 0.0),
                "observed_after": asdict(after),
                "inside_interval": after.low <= predicted <= after.high,
            }
        )
        logger.info(
            "Window %d-%d: shift %.3f (%s) predicts %.3e on %d-%d, observed %.3e [%.3e, %.3e]",
            *window.years,
            decided["shift"],
            decided["chosen"],
            predicted,
            test_first,
            test_last,
            after.rate,
            after.low,
            after.high,
        )
    return rows
