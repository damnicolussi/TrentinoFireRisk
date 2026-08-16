"""The Trentino ignition model: XGBoost tuned on year-blocked CV, and the baselines beside it."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.evaluation import blocked_folds, scores
from tfire.features.registry import Registry, load_registry
from tfire.models.baselines import logistic, random_forest
from tfire.models.mesogeos import PROB_COLUMN

if TYPE_CHECKING:
    from optuna.trial import Trial

logger = logging.getLogger(__name__)

MODEL_FILENAME: Final = "model.json"
METRICS_FILENAME: Final = "metrics.json"

FWI_COLUMN: Final = "fwi"

# Search bounds for the XGBoost hyperparameters.
SEARCH_SPACE: Final[dict[str, tuple[float, float]]] = {
    "max_depth": (3, 10),
    "min_child_weight": (1, 20),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "gamma": (0.0, 5.0),
    "reg_lambda": (0.1, 20.0),
}
INTEGER_PARAMS: Final = frozenset({"max_depth", "min_child_weight"})
LOG_SCALE: Final = frozenset({"learning_rate", "reg_lambda"})

class Estimator(Protocol):
    """What the training loop needs from a model, whether it is XGBoost or an sklearn pipeline."""

    def fit(
        self, features: pd.DataFrame, labels: npt.NDArray[np.int8], **kwargs: object
    ) -> Estimator: ...

    def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]: ...

    def set_params(self, **params: object) -> Estimator: ...


class BoostedEstimator(Estimator, Protocol):
    best_iteration: int

    def save_model(self, path: Path) -> None: ...


Builder = Callable[[Config, float], Estimator]
Selector = Callable[[Sequence[str]], list[str]]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Builder
    columns: Selector
    tuned: bool = False
    early_stopping: bool = False


def _all_columns(names: Sequence[str]) -> list[str]:
    return list(names)


def _without_stacking(names: Sequence[str]) -> list[str]:
    return [name for name in names if name != PROB_COLUMN]


def _fwi_only(names: Sequence[str]) -> list[str]:
    if FWI_COLUMN not in names:
        raise ValueError(f"{FWI_COLUMN} is not in the design matrix")
    return [FWI_COLUMN]


def boosted_trees(config: Config, weight: float) -> Estimator:
    from xgboost import XGBClassifier

    estimator: Estimator = XGBClassifier(
        n_estimators=config.trentino.max_estimators,
        early_stopping_rounds=config.trentino.early_stopping_rounds,
        eval_metric="aucpr",
        scale_pos_weight=weight,
        random_state=config.project.random_seed,
        n_jobs=-1,
    )
    return estimator


SPECS: Final[dict[str, ModelSpec]] = {
    spec.name: spec
    for spec in (
        ModelSpec("xgboost", boosted_trees, _all_columns, tuned=True, early_stopping=True),
        ModelSpec(
            "xgboost_no_stacking",
            boosted_trees,
            _without_stacking,
            tuned=True,
            early_stopping=True,
        ),
        ModelSpec("random_forest", random_forest, _all_columns),
        ModelSpec("logistic", logistic, _all_columns),
        ModelSpec("fwi_only", logistic, _fwi_only),
    )
}


def design_matrix(
    frame: pd.DataFrame, registry: Registry
) -> tuple[pd.DataFrame, npt.NDArray[np.int8], pd.Series]:
    """Features, labels and years off the assembled table."""
    blocks = []
    for spec in registry.present(frame):
        if spec.dtype == "category":
            blocks.append(pd.get_dummies(frame[spec.name], prefix=spec.name, dtype="float32"))
        else:
            blocks.append(frame[[spec.name]].astype("float32"))

    features = pd.concat(blocks, axis=1)
    labels = frame["is_fire"].to_numpy("int8")
    years = frame["date"].dt.year
    logger.info("Design matrix: %d rows x %d columns", len(features), features.shape[1])
    return features, labels, years


def training_mask(years: pd.Series, config: Config) -> npt.NDArray[np.bool_]:
    """Rows the search and every fit are allowed to see. Everything later is scored once."""
    mask: npt.NDArray[np.bool_] = (years < config.trentino.test_years_start).to_numpy()
    if not mask.any() or mask.all():
        raise ValueError(
            f"test_years_start {config.trentino.test_years_start} splits nothing off the record"
        )
    return mask


def positive_weight(labels: npt.NDArray[np.int8]) -> float:
    positives = int(labels.sum())
    if not positives:
        raise ValueError("no positive samples in the split")
    return float(len(labels) - positives) / positives


def _fit(
    spec: ModelSpec,
    estimator: Estimator,
    train: tuple[pd.DataFrame, npt.NDArray[np.int8]],
    validation: tuple[pd.DataFrame, npt.NDArray[np.int8]] | None,
) -> Estimator:
    if spec.early_stopping and validation is not None:
        estimator.fit(*train, eval_set=[validation], verbose=False)
    else:
        estimator.fit(*train)
    return estimator


def cross_validate(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    config: Config,
    params: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], list[int], list[dict[str, Any]]]:
    """Out-of-fold probabilities over the contiguous year blocks, plus each fold's own scores."""
    columns = spec.columns(list(features.columns))
    out_of_fold = np.full(len(labels), np.nan)
    rounds: list[int] = []
    fold_scores: list[dict[str, Any]] = []

    for train_index, validation_index in blocked_folds(years, config.trentino.cv_folds):
        train = (features.iloc[train_index][columns], labels[train_index])
        validation = (features.iloc[validation_index][columns], labels[validation_index])

        estimator = spec.build(config, positive_weight(train[1]))
        if params:
            estimator.set_params(**params)
        _fit(spec, estimator, train, validation)
        if spec.early_stopping:
            rounds.append(cast("BoostedEstimator", estimator).best_iteration + 1)

        probabilities = estimator.predict_proba(validation[0])[:, 1]
        out_of_fold[validation_index] = probabilities
        held = years.iloc[validation_index]
        fold_scores.append(
            {
                "years": f"{held.min()}-{held.max()}",
                "rows": len(held),
                **scores(validation[1], probabilities),
            }
        )

    return out_of_fold, rounds, fold_scores


def _suggest(trial: Trial) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, (low, high) in SEARCH_SPACE.items():
        log = name in LOG_SCALE
        if name in INTEGER_PARAMS:
            params[name] = trial.suggest_int(name, int(low), int(high), log=log)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log)
    return params


def tune(
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    config: Config,
) -> dict[str, Any]:
    """Optuna over the XGBoost hyperparameters, maximizing the pooled out-of-fold AUPRC."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = SPECS["xgboost"]

    def objective(trial: Trial) -> float:
        params = _suggest(trial)
        out_of_fold, rounds, _ = cross_validate(spec, features, labels, years, config, params)
        trial.set_user_attr("rounds", int(np.median(rounds)))
        return scores(labels, out_of_fold)["auprc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.project.random_seed),
    )
    study.optimize(objective, n_trials=config.trentino.optuna_trials, n_jobs=1)

    best = study.best_trial
    logger.info(
        "Optuna: %d trials, best pooled out-of-fold AUPRC %.4f at %d rounds",
        len(study.trials),
        study.best_value,
        best.user_attrs["rounds"],
    )
    return {
        "trials": len(study.trials),
        "best_trial": best.number,
        "best_auprc": float(study.best_value),
        "rounds": int(best.user_attrs["rounds"]),
        "params": dict(best.params),
        "search_space": {name: list(bounds) for name, bounds in SEARCH_SPACE.items()},
    }


def _final_params(tuning: dict[str, Any]) -> dict[str, Any]:
    """The tuned settings with the search's early stopping replaced by the settled count."""
    return {**tuning["params"], "n_estimators": tuning["rounds"], "early_stopping_rounds": None}


def evaluate_spec(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
) -> dict[str, Any]:
    """Year-blocked CV inside the training span, then one fit scored on the held-out years."""
    params = _final_params(tuning) if spec.tuned else None
    columns = spec.columns(list(features.columns))

    out_of_fold, _, fold_scores = cross_validate(
        spec,
        features.loc[train],
        labels[train],
        years.loc[train],
        config,
        tuning["params"] if spec.tuned else None,
    )
    pooled = scores(labels[train], out_of_fold)

    estimator = spec.build(config, positive_weight(labels[train]))
    if params:
        estimator.set_params(**params)
    _fit(
        spec,
        estimator,
        (features.loc[train][columns], labels[train]),
        None,
    )
    held = scores(labels[~train], estimator.predict_proba(features.loc[~train][columns])[:, 1])

    logger.info(
        "%-20s | pooled AUPRC %.4f (lift %.2f) | holdout AUPRC %.4f (lift %.2f)",
        spec.name,
        pooled["auprc"],
        pooled["lift"],
        held["auprc"],
        held["lift"],
    )
    return {
        "features": len(columns),
        "excluded": [name for name in features.columns if name not in set(columns)],
        "folds": fold_scores,
        "pooled_out_of_fold": pooled,
        "holdout": held,
        "hyperparameters": params if spec.tuned else "fixed, see config",
    }


def _ship(
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    config: Config,
    tuning: dict[str, Any],
    path: Path,
) -> None:
    """Refit the tuned model on the whole record, which is what inference will load."""
    spec = SPECS["xgboost"]
    estimator = spec.build(config, positive_weight(labels))
    estimator.set_params(**_final_params(tuning))
    estimator.fit(features[spec.columns(list(features.columns))], labels, verbose=False)
    cast("BoostedEstimator", estimator).save_model(path)


def train_trentino(
    config: Config, force: bool = False, selected: Sequence[str] | None = None
) -> dict[str, Any]:
    """Train the selected models and persist them under `models/trentino/<version>/`."""
    directory = config.path(config.paths.trentino_model_dir) / config.trentino.version
    metrics_path = directory / METRICS_FILENAME
    stored: dict[str, Any] = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    )
    if stored and not force:
        logger.info("Metrics already exist, skipping (use --force to retrain): %s", metrics_path)
        return stored

    names = list(SPECS) if not selected else list(selected)
    specs = [SPECS[name] for name in names]

    frame = pd.read_parquet(config.path(config.paths.dataset_out))
    features, labels, years = design_matrix(frame, load_registry())

    train = training_mask(years, config)
    logger.info(
        "Train %d-%d: %d rows, %.2f%% positive | holdout %d-%d: %d rows, %.2f%% positive",
        years[train].min(),
        years[train].max(),
        int(train.sum()),
        100 * labels[train].mean(),
        years[~train].min(),
        years[~train].max(),
        int((~train).sum()),
        100 * labels[~train].mean(),
    )

    if any(spec.tuned for spec in specs):
        tuning = tune(features.loc[train], labels[train], years.loc[train], config)
    else:
        tuning = stored.get("tuning", {})
        if not tuning:
            raise ValueError("No stored tuning to reuse: run with the xgboost model selected")

    models = dict(stored.get("models", {}))
    for spec in specs:
        models[spec.name] = evaluate_spec(spec, features, labels, years, train, config, tuning)

    metrics = {
        "version": config.trentino.version,
        "split": {
            "train_years": [int(years[train].min()), int(years[train].max())],
            "test_years": [int(years[~train].min()), int(years[~train].max())],
            "cv_folds": config.trentino.cv_folds,
            "rows": {"train": int(train.sum()), "holdout": int((~train).sum())},
        },
        "tuning": tuning,
        "models": models,
        "columns": list(features.columns),
        "random_seed": config.project.random_seed,
        "config_sha256": config.digest(),
    }

    directory.mkdir(parents=True, exist_ok=True)
    if "xgboost" in names:
        _ship(features, labels, config, tuning, directory / MODEL_FILENAME)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %d model(s) and their metrics to %s", len(specs), directory)
    return metrics
