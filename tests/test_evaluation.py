"""The evaluation metrics, the spatial blocking and the case-control correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.evaluation import (
    block_index,
    bootstrap_scores,
    calibration_bins,
    expected_calibration_error,
    precision_at_k,
    scores,
    spatial_folds,
)
from tfire.models.calibration import Calibrator


def test_precision_at_k_counts_the_top_of_the_ranking() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    rows = precision_at_k(labels, probabilities, [0.2, 0.5])

    assert [row["k"] for row in rows] == [2, 5]
    assert [row["caught"] for row in rows] == [1, 3]
    assert [row["precision"] for row in rows] == [0.5, 0.6]
    assert [row["recall"] for row in rows] == [1 / 3, 1.0]
    assert [row["threshold"] for row in rows] == [0.8, 0.5]


def test_calibration_bins_hold_every_row_once() -> None:
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.1, 0.15, 0.2, 0.6, 0.7, 0.8, 0.9])

    bins = calibration_bins(labels, probabilities, 2)

    assert sum(row["count"] for row in bins) == len(labels)
    assert [row["observed"] for row in bins] == [0.25, 0.75]
    assert expected_calibration_error(bins) == pytest.approx(
        (4 * abs(0.125 - 0.25) + 4 * abs(0.75 - 0.75)) / 8
    )


def test_calibration_bins_survive_a_probability_repeated_across_quantiles() -> None:
    """A model that predicts one value for most rows collapses the quantile edges."""
    labels = np.array([0] * 9 + [1])
    probabilities = np.array([0.02] * 9 + [0.8])

    bins = calibration_bins(labels, probabilities, 10)

    assert sum(row["count"] for row in bins) == len(labels)


def test_spatial_folds_hold_out_whole_blocks() -> None:
    x = pd.Series(np.repeat(np.arange(0, 60000, 10000, dtype="float64"), 4))
    y = pd.Series(np.tile(np.arange(0, 40000, 10000, dtype="float64"), 6))
    blocks = block_index(x, y, 20000)

    seen: list[int] = []
    for train_index, validation_index in spatial_folds(x, y, 20000, 3, seed=42):
        assert not set(train_index) & set(validation_index)
        assert len(train_index) + len(validation_index) == len(x)
        held = set(blocks[validation_index])
        assert not held & set(blocks[train_index])
        seen.extend(validation_index.tolist())

    assert sorted(seen) == list(range(len(x)))


def test_prior_correction_recovers_the_population_rate() -> None:
    """A case-control sample with a known population rate, mapped back to it."""
    population_negatives = 1_000_000
    sampled_negatives = 10_000
    rate = sampled_negatives / population_negatives

    calibrator = Calibrator(
        thresholds=[0.0, 1.0],
        values=[0.0, 1.0],
        log_offset=float(np.log(rate)),
        sampling_rate=rate,
        counts={},
    )
    positives = 500
    sample_rate = positives / (positives + sampled_negatives)
    expected = positives / (positives + population_negatives)

    corrected = calibrator.to_population_rate(np.array([sample_rate]))
    assert float(corrected[0]) == pytest.approx(expected, rel=1e-6)


def test_isotonic_never_inverts_two_predictions(config: Config) -> None:
    """The calibrator may tie two rows together, never swap them.

    Isotonic is monotone but not strictly so, which is why this is not an assertion that AUROC
    comes out unchanged: flattening a region merges rows into ties and moves it slightly.
    """
    from tfire.models.calibration import isotonic_knots

    rng = np.random.default_rng(config.project.random_seed)
    labels = rng.binomial(1, 0.1, size=2000)
    probabilities = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.2, size=2000), 0.001, 0.999)

    thresholds, values = isotonic_knots(labels, probabilities)
    calibrator = Calibrator(thresholds, values, log_offset=0.0, sampling_rate=1.0, counts={})
    recalibrated = calibrator.to_sample_rate(probabilities)

    ordered = recalibrated[np.argsort(probabilities, kind="stable")]
    assert np.all(np.diff(ordered) >= 0)
    assert float(recalibrated.mean()) == pytest.approx(float(labels.mean()), abs=0.01)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 0], [40, 360])
    probabilities = 1 / (1 + np.exp(-rng.normal(labels * 1.2 - 1.0, 1.0)))

    point = scores(labels, probabilities)
    interval = bootstrap_scores(labels, probabilities, 200, seed=0)

    for metric in ("auprc", "auroc", "lift"):
        assert interval[metric]["lo"] < point[metric] < interval[metric]["hi"]
        assert interval[metric]["resamples"] == 200


def test_bootstrap_discards_resamples_with_one_class() -> None:
    labels = np.array([1] + [0] * 19)
    probabilities = np.linspace(1.0, 0.0, 20)

    interval = bootstrap_scores(labels, probabilities, 100, seed=0)

    assert 0 < interval["auprc"]["resamples"] < 100
