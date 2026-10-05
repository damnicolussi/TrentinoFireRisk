"""The Trentino ignition model: XGBoost tuned on year-blocked CV, and the baselines beside it."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection, Sequence
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
HOLDOUT_MODEL_FILENAME: Final = "model_holdout.json"
METRICS_FILENAME: Final = "metrics.json"
TRIALS_FILENAME: Final = "tuning_trials.json"

FWI_COLUMN: Final = "fwi"

# the one categorical feature, and so the one whose indicators a single-day matrix can lack
_CATEGORICAL_PREFIX: Final = "season"

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

# v2's search ended on the lower bound of colsample_bytree, with all ten of its best trials under
# 0.5; v3 opens the bound and adds sampling per split
SEARCH_SPACES: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "v2": SEARCH_SPACE,
    "v3": {**SEARCH_SPACE, "colsample_bytree": (0.1, 1.0), "colsample_bynode": (0.1, 1.0)},
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


def align_columns(
    features: pd.DataFrame, columns: Sequence[str], declared: Collection[str] = ()
) -> pd.DataFrame:
    """Put a design matrix into the exact column set and order a stored model expects.

    A single day carries one season, so the categorical expands to one indicator instead of
    four. The absent indicators are genuinely zero and are filled; anything else missing is a
    broken assembly and raises rather than being scored as a silent zero. A `declared` feature
    the model was fitted without is dropped, so an older model keeps serving after the registry
    grows; any other extra column still raises.
    """
    missing = [name for name in columns if name not in features.columns]
    invented = [name for name in missing if not name.startswith(f"{_CATEGORICAL_PREFIX}_")]
    if invented:
        raise ValueError(f"The design matrix is missing {len(invented)} column(s): {invented}")

    extra = [name for name in features.columns if name not in set(columns) | set(declared)]
    if extra:
        raise ValueError(f"The design matrix carries {len(extra)} undeclared column(s): {extra}")

    aligned = features.reindex(columns=list(columns))
    if missing:
        aligned[missing] = 0.0
    return aligned.astype("float32")


def training_mask(years: pd.Series, config: Config) -> npt.NDArray[np.bool_]:
    """Rows the search and every fit are allowed to see. Everything later is scored once."""
    mask: npt.NDArray[np.bool_] = (years < config.trentino.test_years_start).to_numpy()
    if not mask.any() or mask.all():
        raise ValueError(
            f"test_years_start {config.trentino.test_years_start} splits nothing off the record"
        )
    return mask


def positive_weight(
    labels: npt.NDArray[np.int8], weights: npt.NDArray[np.float64] | None = None
) -> float:
    """Negatives over positives, each row counted at its sample weight when there is one."""
    positives = int(labels.sum())
    if not positives:
        raise ValueError("no positive samples in the split")
    if weights is None:
        return float(len(labels) - positives) / positives
    positive = labels == 1
    return float(weights[~positive].sum() / weights[positive].sum())


def _fit(
    spec: ModelSpec,
    estimator: Estimator,
    train: tuple[pd.DataFrame, npt.NDArray[np.int8]],
    validation: tuple[pd.DataFrame, npt.NDArray[np.int8]] | None,
    weights: npt.NDArray[np.float64] | None = None,
) -> Estimator:
    features, labels = train
    if spec.early_stopping and validation is not None:
        if weights is None:
            estimator.fit(features, labels, eval_set=[validation], verbose=False)
        else:
            estimator.fit(
                features, labels, eval_set=[validation], verbose=False, sample_weight=weights
            )
    elif weights is None:
        estimator.fit(features, labels)
    else:
        estimator.fit(features, labels, sample_weight=weights)
    return estimator


def cross_validate(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    config: Config,
    params: dict[str, float] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], list[int], list[dict[str, Any]]]:
    """Out-of-fold probabilities over the contiguous year blocks, plus each fold's own scores.

    `weights`, one per row, weigh the training rows only; a held-out block is scored as it is.
    """
    columns = spec.columns(list(features.columns))
    out_of_fold = np.full(len(labels), np.nan)
    rounds: list[int] = []
    fold_scores: list[dict[str, Any]] = []

    for train_index, validation_index in blocked_folds(years, config.trentino.cv_folds):
        train = (features.iloc[train_index][columns], labels[train_index])
        validation = (features.iloc[validation_index][columns], labels[validation_index])

        fold_weights = None if weights is None else weights[train_index]
        estimator = spec.build(config, positive_weight(train[1], fold_weights))
        if params:
            estimator.set_params(**params)
        _fit(spec, estimator, train, validation, fold_weights)
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
                "balanced_log_loss": balanced_log_loss(validation[1], probabilities),
            }
        )

    return out_of_fold, rounds, fold_scores


def balanced_log_loss(labels: npt.NDArray[Any], probabilities: npt.NDArray[Any]) -> float:
    """Log-loss with each class weighted to one half.

    The fits carry `scale_pos_weight`, so their outputs sit on a balanced scale. Plain log-loss
    would score them against each fold's own prevalence, which runs from 16% to 3%.
    """
    clipped = np.clip(np.asarray(probabilities, dtype="float64"), 1e-15, 1 - 1e-15)
    positive = np.asarray(labels) == 1
    return float(-0.5 * (np.log(clipped[positive]).mean() + np.log1p(-clipped[~positive]).mean()))


def _fold_mean(folds: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([fold[key] for fold in folds]))


# what a trial is ranked by, larger always better. Pooled AUPRC weighs the early folds, where
# positives are five times as frequent, so the fold-level means are the base-rate robust ones.
Objective = Callable[[float, Sequence[dict[str, Any]]], float]
OBJECTIVES: Final[dict[str, Objective]] = {
    "pooled_auprc": lambda pooled, folds: pooled,
    "fold_lift": lambda pooled, folds: _fold_mean(folds, "lift"),
    "fold_logloss": lambda pooled, folds: -_fold_mean(folds, "balanced_log_loss"),
    "fold_auroc": lambda pooled, folds: _fold_mean(folds, "auroc"),
}


def _suggest(trial: Trial, space: dict[str, tuple[float, float]]) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, (low, high) in space.items():
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
    """Optuna over the XGBoost hyperparameters, maximizing `trentino.tuning_objective`.

    Every trial keeps its pooled AUPRC and its per-fold scores, so the same trials can be
    ranked again under another objective without refitting anything.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = SPECS["xgboost"]
    ranking = OBJECTIVES[config.trentino.tuning_objective]
    space = SEARCH_SPACES[config.trentino.search_space]

    def objective(trial: Trial) -> float:
        params = _suggest(trial, space)
        out_of_fold, rounds, folds = cross_validate(spec, features, labels, years, config, params)
        pooled = scores(labels, out_of_fold)["auprc"]
        trial.set_user_attr("rounds", int(np.median(rounds)))
        trial.set_user_attr("pooled_auprc", pooled)
        trial.set_user_attr("folds", folds)
        if trial.number % 10 == 9:
            logger.info("Optuna: trial %d done", trial.number + 1)
        return ranking(pooled, folds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.project.random_seed),
    )
    study.optimize(objective, n_trials=config.trentino.optuna_trials, n_jobs=1)

    best = study.best_trial
    logger.info(
        "Optuna: %d trials, best %s %.4f (pooled AUPRC %.4f) at %d rounds",
        len(study.trials),
        config.trentino.tuning_objective,
        study.best_value,
        best.user_attrs["pooled_auprc"],
        best.user_attrs["rounds"],
    )
    return {
        "trials": len(study.trials),
        "objective": config.trentino.tuning_objective,
        "best_trial": best.number,
        "best_value": float(study.best_value),
        "best_auprc": float(best.user_attrs["pooled_auprc"]),
        "rounds": int(best.user_attrs["rounds"]),
        "params": dict(best.params),
        "search_space": {name: list(bounds) for name, bounds in space.items()},
        "boundary": on_the_boundary(dict(best.params), space),
        "trial_table": [
            {
                "number": trial.number,
                "params": dict(trial.params),
                "rounds": int(trial.user_attrs["rounds"]),
                "pooled_auprc": float(trial.user_attrs["pooled_auprc"]),
                "folds": trial.user_attrs["folds"],
            }
            for trial in study.trials
        ],
    }


def on_the_boundary(
    params: dict[str, float], space: dict[str, tuple[float, float]], margin: float = 0.02
) -> list[str]:
    """Parameters the best trial left within `margin` of the span from either bound."""
    hits = []
    for name, (low, high) in space.items():
        if name in params:
            reach = margin * (high - low)
            if params[name] - low <= reach or high - params[name] <= reach:
                hits.append(name)
    return hits


def final_params(tuning: dict[str, Any]) -> dict[str, Any]:
    """The tuned settings with the search's early stopping replaced by the settled count."""
    return {**tuning["params"], "n_estimators": tuning["rounds"], "early_stopping_rounds": None}


@dataclass(frozen=True)
class Fitted:
    """A model fitted on the training years"""

    estimator: Estimator
    columns: list[str]
    out_of_fold: npt.NDArray[np.float64]
    holdout: npt.NDArray[np.float64]
    fold_scores: list[dict[str, Any]]


def fit_holdout(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
    weights: npt.NDArray[np.float64] | None = None,
) -> tuple[Estimator, list[str], npt.NDArray[np.float64]]:
    """One fit on the training years and its probabilities over the held-out ones."""
    columns = spec.columns(list(features.columns))
    train_weights = None if weights is None else weights[train]
    estimator = spec.build(config, positive_weight(labels[train], train_weights))
    if spec.tuned:
        estimator.set_params(**final_params(tuning))
    _fit(spec, estimator, (features.loc[train][columns], labels[train]), None, train_weights)

    holdout = estimator.predict_proba(features.loc[~train][columns])[:, 1]
    return estimator, columns, holdout


def fit_and_predict(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
) -> Fitted:
    """Year-blocked CV inside the training span, then one fit scoring the held-out years."""
    out_of_fold, _, fold_scores = cross_validate(
        spec,
        features.loc[train],
        labels[train],
        years.loc[train],
        config,
        tuning["params"] if spec.tuned else None,
    )
    estimator, columns, holdout = fit_holdout(spec, features, labels, train, config, tuning)
    return Fitted(
        estimator=estimator,
        columns=columns,
        out_of_fold=out_of_fold,
        holdout=holdout,
        fold_scores=fold_scores,
    )


def evaluate_spec(
    spec: ModelSpec,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    train: npt.NDArray[np.bool_],
    config: Config,
    tuning: dict[str, Any],
) -> tuple[dict[str, Any], Estimator]:
    """One model's fold, pooled out-of-fold and holdout scores, and its holdout fit."""
    fitted = fit_and_predict(spec, features, labels, years, train, config, tuning)
    pooled = scores(labels[train], fitted.out_of_fold)
    held = scores(labels[~train], fitted.holdout)

    logger.info(
        "%-20s | pooled AUPRC %.4f (lift %.2f) | holdout AUPRC %.4f (lift %.2f)",
        spec.name,
        pooled["auprc"],
        pooled["lift"],
        held["auprc"],
        held["lift"],
    )
    return {
        "features": len(fitted.columns),
        "excluded": [name for name in features.columns if name not in set(fitted.columns)],
        "folds": fitted.fold_scores,
        "pooled_out_of_fold": pooled,
        "holdout": held,
        "hyperparameters": final_params(tuning) if spec.tuned else "fixed, see config",
    }, fitted.estimator


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
    estimator.set_params(**final_params(tuning))
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

    trials: list[dict[str, Any]] = []
    if any(spec.tuned for spec in specs):
        tuning = tune(features.loc[train], labels[train], years.loc[train], config)
        trials = tuning.pop("trial_table")
    else:
        tuning = stored.get("tuning", {})
        if not tuning:
            raise ValueError("No stored tuning to reuse: run with the xgboost model selected")

    models = dict(stored.get("models", {}))
    holdout_fit: Estimator | None = None
    for spec in specs:
        models[spec.name], estimator = evaluate_spec(
            spec, features, labels, years, train, config, tuning
        )
        if spec.name == "xgboost":
            holdout_fit = estimator

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
    if holdout_fit is not None:
        _ship(features, labels, config, tuning, directory / MODEL_FILENAME)
        cast("BoostedEstimator", holdout_fit).save_model(directory / HOLDOUT_MODEL_FILENAME)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if trials:
        (directory / TRIALS_FILENAME).write_text(json.dumps(trials, indent=2), encoding="utf-8")
    logger.info("Wrote %d model(s) and their metrics to %s", len(specs), directory)
    return metrics


def save_holdout_model(config: Config) -> Path:
    """Refit the stored xgboost settings on the training years alone and save that fit."""
    from tfire.models.evaluate import check_reproducible

    directory = config.path(config.paths.trentino_model_dir) / config.trentino.version
    metrics_path = directory / METRICS_FILENAME
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No trained model at {metrics_path}. Run `tfire train` first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    frame = pd.read_parquet(config.path(config.paths.dataset_out))
    features, labels, years = design_matrix(frame, load_registry())
    if list(features.columns) != list(metrics["columns"]):
        raise ValueError("The assembled table no longer has the columns the stored model used")
    train = training_mask(years, config)

    estimator, _, holdout = fit_holdout(
        SPECS["xgboost"], features, labels, train, config, metrics["tuning"]
    )
    check_reproducible(metrics["models"]["xgboost"]["holdout"], scores(labels[~train], holdout))

    path = directory / HOLDOUT_MODEL_FILENAME
    cast("BoostedEstimator", estimator).save_model(path)
    logger.info("Wrote the %d-%d fit to %s", years[train].min(), years[train].max(), path)
    return path
