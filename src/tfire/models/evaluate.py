"""Scoring, attribution, calibration and sensitivity runs for the trained model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire import figures, sensitivity
from tfire.config import Config
from tfire.evaluation import precision_at_k, scores, spatial_folds
from tfire.features.registry import load_registry
from tfire.models import calibration
from tfire.models.explain import attribute
from tfire.models.trentino import (
    METRICS_FILENAME,
    SPECS,
    ModelSpec,
    design_matrix,
    final_params,
    fit_and_predict,
    fit_holdout,
    positive_weight,
    training_mask,
)
from tfire.report import render_report

logger = logging.getLogger(__name__)

EVALUATION_FILENAME = "evaluation.json"
PRIMARY = "xgboost"

# past this, a refit of the stored hyperparameters is not reproducing the stored scores
_REPRODUCIBILITY_TOLERANCE = 1e-6


def load_metrics(config: Config) -> tuple[Path, dict[str, Any]]:
    directory = config.path(config.paths.trentino_model_dir) / config.trentino.version
    path = directory / METRICS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}. Run `tfire train` first.")
    return directory, json.loads(path.read_text(encoding="utf-8"))


def check_reproducible(stored: dict[str, Any], recomputed: dict[str, float]) -> dict[str, float]:
    """Refuse to report an evaluation of a model that no longer scores what was shipped."""
    drift = {
        metric: abs(recomputed[metric] - stored[metric])
        for metric in ("auprc", "auroc", "brier")
        if abs(recomputed[metric] - stored[metric]) > _REPRODUCIBILITY_TOLERANCE
    }
    if drift:
        raise ValueError(
            f"Refitting the stored hyperparameters does not reproduce {METRICS_FILENAME}: "
            + ", ".join(
                f"{metric} {stored[metric]:.6f} -> {recomputed[metric]:.6f}" for metric in drift
            )
        )
    logger.info(
        "Recomputed holdout scores match the stored ones to %.0e", _REPRODUCIBILITY_TOLERANCE
    )
    return recomputed


def spatial_cross_validation(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    frame: pd.DataFrame,
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
) -> dict[str, Any]:
    """The year-blocked CV's rows and model, partitioned by spatial block instead.

    Negatives sit in different cells from the positives, so holding whole blocks out is what
    separates knowing which cells burn from knowing which day burns.
    """
    columns = spec.columns(list(features.columns))
    subset = features.loc[train]
    coordinates = frame.loc[train]
    held_labels = labels[train]
    out_of_fold = np.full(len(held_labels), np.nan)
    fold_scores = []

    folds = spatial_folds(
        coordinates["x_coordinate"],
        coordinates["y_coordinate"],
        config.evaluation.spatial_block_m,
        config.evaluation.spatial_folds,
        config.project.random_seed,
    )
    for fold, (train_index, validation_index) in enumerate(folds):
        estimator = spec.build(config, positive_weight(held_labels[train_index]))
        estimator.set_params(**final_params(tuning))
        estimator.fit(subset.iloc[train_index][columns], held_labels[train_index], verbose=False)

        probabilities = estimator.predict_proba(subset.iloc[validation_index][columns])[:, 1]
        out_of_fold[validation_index] = probabilities
        fold_scores.append(
            {
                "fold": fold,
                "rows": len(validation_index),
                **scores(held_labels[validation_index], probabilities),
            }
        )

    pooled = scores(held_labels, out_of_fold)
    logger.info(
        "Spatially blocked CV over %d block group(s): pooled AUPRC %.4f, AUROC %.4f",
        config.evaluation.spatial_folds,
        pooled["auprc"],
        pooled["auroc"],
    )
    return {
        "block_m": config.evaluation.spatial_block_m,
        "folds": fold_scores,
        "pooled_out_of_fold": pooled,
    }


def baseline_probabilities(
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
    primary: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[Any]]:
    """Holdout probabilities for every model, for the curves. No CV: the scores are stored."""
    probabilities = {PRIMARY: primary}
    for name, spec in SPECS.items():
        if name == PRIMARY:
            continue
        _, _, holdout = fit_holdout(spec, features, labels, train, config, tuning)
        probabilities[name] = holdout
    return probabilities


def evaluate_trentino(
    config: Config, force: bool = False, variants: list[str] | None = None
) -> dict[str, Any]:
    """Score, attribute, calibrate and stress the shipped model, then write the figures."""
    directory, metrics = load_metrics(config)
    out = directory / EVALUATION_FILENAME
    if out.is_file() and not force:
        logger.info("Evaluation already exists, skipping (use --force to rerun): %s", out)
        return dict(json.loads(out.read_text(encoding="utf-8")))

    tuning = metrics["tuning"]
    registry = load_registry()
    frame = pd.read_parquet(config.path(config.paths.dataset_out))
    features, labels, years = design_matrix(frame, registry)
    train = training_mask(years, config)

    spec = SPECS[PRIMARY]
    fitted = fit_and_predict(spec, features, labels, years, train, config, tuning)
    holdout = check_reproducible(
        metrics["models"][PRIMARY]["holdout"], scores(labels[~train], fitted.holdout)
    )
    pooled = scores(labels[train], fitted.out_of_fold)

    probabilities = baseline_probabilities(features, labels, train, config, tuning, fitted.holdout)
    spatial = spatial_cross_validation(spec, features, labels, frame, train, config, tuning)

    calibrator = calibration.fit(
        labels[train], fitted.out_of_fold, config, int((labels == 0).sum())
    )
    reliability = calibration.report(
        labels[~train], fitted.holdout, calibrator, config.evaluation.calibration_bins
    )
    explained = attribute(fitted.estimator, features.loc[~train][fitted.columns], registry, frame)

    results: dict[str, Any] = {}
    if variants:
        results = sensitivity.run(
            config, sensitivity.resolve(variants), tuning, metrics["models"][PRIMARY]
        )

    evaluation = {
        "version": config.trentino.version,
        "rows": {"train": int(train.sum()), "holdout": int((~train).sum())},
        "pooled_out_of_fold": pooled,
        "holdout": holdout,
        "folds": fitted.fold_scores,
        "precision_at_k": {
            "pooled_out_of_fold": precision_at_k(
                labels[train], fitted.out_of_fold, config.evaluation.precision_at_k
            ),
            "holdout": precision_at_k(
                labels[~train], fitted.holdout, config.evaluation.precision_at_k
            ),
        },
        "spatial_cv": spatial,
        "calibration": reliability,
        "sampling_correction": {
            "log_offset": calibrator.log_offset,
            "sampling_rate": calibrator.sampling_rate,
            **calibrator.counts,
        },
        "attribution": {
            "rows": len(features.loc[~train]),
            "per_feature": explained.per_feature.to_dict("records"),
            "per_category": explained.per_category.to_dict("records"),
            "per_temporal": explained.per_temporal.to_dict("records"),
        },
        "sensitivity": results,
        "random_seed": config.project.random_seed,
        "config_sha256": config.digest(),
    }

    written = [
        *figures.curves(labels[~train], probabilities, config),
        figures.reliability(reliability, config),
        figures.fold_metrics(fitted.fold_scores, config),
        *figures.attribution(explained, config),
        figures.spatial_comparison(pooled, spatial["pooled_out_of_fold"], config),
    ]
    sensitivity_figure = figures.sensitivity(results, config)
    if sensitivity_figure:
        written.append(sensitivity_figure)

    calibrator.write(directory / calibration.CALIBRATOR_FILENAME)
    out.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    report = render_report(evaluation, metrics, written, config)

    logger.info("Wrote %d figure(s) to %s", len(written), config.path(config.paths.figures_dir))
    logger.info("Wrote %s and %s", out, report)
    return evaluation
