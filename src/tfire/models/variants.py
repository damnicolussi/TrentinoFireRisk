"""Retraining variants, each one design choice away from v2, scored on v2's holdout and event axis.

Variants keep v2's tuned hyperparameters unless they ask for their own search, so a gap between
two of them is the design and not Optuna.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.evaluation import bootstrap_scores, scores
from tfire.features.registry import load_registry
from tfire.models.trentino import (
    HOLDOUT_MODEL_FILENAME,
    METRICS_FILENAME,
    SPECS,
    BoostedEstimator,
    Estimator,
    align_columns,
    cross_validate,
    design_matrix,
    fit_holdout,
    training_mask,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

EXPERIMENTS_DIR: Final = "experiments"
VARIANT_FILENAME: Final = "variant.json"
TUNING_FILENAME: Final = "tuning.json"
RESULTS_FILENAME: Final = "results.json"
REFERENCE_VERSION: Final = "v2"
REFERENCE_NAME: Final = "v2"

# the expected ignitions along a road are summed over one holdout day in this many
MASS_STRIDE_DAYS: Final = 7

# a variant passes when its holdout AUPRC is no more than this below v2's and its median
# ignition percentile is not lower
AUPRC_TOLERANCE: Final = 0.02

_LOGIT_CLIP: Final = 1e-7

# CORINE columns by level: 44 third-level classes, then the second- and first-level aggregates
_CLC_LEVELS: Final = {
    "l3": re.compile(r"CLC_\d+$"),
    "l2": re.compile(r"CLC_L2_"),
    "l1": re.compile(r"CLC_L1_"),
}


@dataclass(frozen=True)
class Variant:
    """One design choice away from v2, everything else held at v2's settings."""

    name: str
    question: str
    half_life: float | None = None
    first_year: int | None = None
    columns: Callable[[Sequence[str]], list[str]] | None = None
    # set either, and the variant runs its own search on its training years instead of
    # borrowing v2's hyperparameters
    objective: str | None = None
    search_space: str | None = None

    @property
    def retunes(self) -> bool:
        return self.objective is not None or self.search_space is not None


def without_land_cover(*levels: str) -> Callable[[Sequence[str]], list[str]]:
    """A column selector that drops the CORINE columns of the given levels."""
    patterns = [_CLC_LEVELS[level] for level in levels]
    return lambda columns: [c for c in columns if not any(p.match(c) for p in patterns)]


def adding(*names: str) -> Callable[[Sequence[str]], list[str]]:
    """A column selector that appends new features after the reference model's own."""
    return lambda columns: [*columns, *(name for name in names if name not in columns)]


# what the lapse correction replaces: every temperature and humidity column with a cell-height
# counterpart. The temperature range does not move with a uniform shift, and the FWI codes stay
# on the backbone cell they accumulate on.
LAPSED: Final = (
    "temp_mean",
    "temp_min",
    "temp_max",
    "temp_noon",
    "temp_mean_3d",
    "rh_mean",
    "rh_min",
    "rh_max",
    "rh_range",
    "rh_noon",
    "rh_mean_7d",
    "vpd_max",
)


def lapsed(*added: str) -> Callable[[Sequence[str]], list[str]]:
    """`adding`, then each backbone temperature and humidity swapped for its cell-height one."""
    return lambda columns: [
        f"{name}_lapse" if name in LAPSED else name for name in adding(*added)(columns)
    ]


VARIANT_SETS: Final[dict[str, tuple[Variant, ...]]] = {
    "recency": (
        Variant(REFERENCE_NAME, "v2 refitted, the reference every other row is read against"),
        Variant("half_life_20", "do recent years deserve more weight, slowly?", half_life=20.0),
        Variant("half_life_10", "the same with a ten-year half-life", half_life=10.0),
        Variant("half_life_5", "the same with a five-year half-life", half_life=5.0),
        Variant("from_1994", "do the earliest ten years only add noise?", first_year=1994),
        Variant("from_2000", "is the pre-2000 regime worth training on at all?", first_year=2000),
    ),
    "land_cover": (
        Variant(REFERENCE_NAME, "v2 refitted, all 64 land-cover columns (44 L3, 15 L2, 5 L1)"),
        Variant(
            "clc_l2_l1",
            "do the 44 third-level classes only pin the map to the cell? (20 columns)",
            columns=without_land_cover("l3"),
        ),
        Variant(
            "clc_l2", "the 15 second-level classes alone", columns=without_land_cover("l3", "l1")
        ),
        Variant(
            "no_land_cover", "no land cover at all", columns=without_land_cover("l3", "l2", "l1")
        ),
    ),
    "search": (
        Variant(REFERENCE_NAME, "v2 refitted, tuned on pooled AUPRC inside the v2 bounds"),
        Variant(
            "retuned",
            "does a search on fold lift, with colsample opened to 0.1 and per split, find more?",
            objective="fold_lift",
            search_space="v3",
        ),
    ),
    # cumulative, each row keeps the features of the one above it. Needs meteo.cell_scale on,
    # and the backbone and the table rebuilt with it
    "cell_meteo": (
        Variant(REFERENCE_NAME, "v2 refitted, the meteorology as shipped"),
        Variant(
            "days_since_rain",
            "does the length of the dry spell add what the 7-30 day sums miss?",
            columns=adding("days_since_rain"),
        ),
        Variant(
            "vpd_max",
            "and the afternoon vapor pressure deficit, derived hourly?",
            columns=adding("days_since_rain", "vpd_max"),
        ),
        Variant(
            "lapse",
            "and temperature and humidity at the cell's own height, not the backbone's?",
            columns=lapsed("days_since_rain", "vpd_max"),
        ),
    ),
}


def recency_weights(
    years: npt.NDArray[np.int64], last_year: int, half_life: float | None
) -> FloatArray | None:
    """`0.5 ** (age / half_life)`, age counted back from `last_year`; None without a half-life."""
    if half_life is None:
        return None
    return np.asarray(np.power(0.5, (last_year - years) / half_life), dtype="float64")


@dataclass(frozen=True)
class Design:
    """The rows and columns one variant trains and is scored on."""

    features: pd.DataFrame
    labels: npt.NDArray[np.int8]
    years: pd.Series
    train: BoolArray
    weights: FloatArray | None


def design(
    variant: Variant,
    features: pd.DataFrame,
    labels: npt.NDArray[np.int8],
    years: pd.Series,
    reference_columns: Sequence[str],
    config: Config,
) -> Design:
    """Drop the training rows before `first_year` and the columns the variant leaves out.

    The holdout rows are never dropped: every variant is scored on the same years. Columns start
    from the reference model's, so a feature added to the table later does not enter a variant
    that did not ask for it.
    """
    columns = list(reference_columns)
    if variant.columns is not None:
        columns = variant.columns(columns)

    keep = np.ones(len(labels), dtype=bool)
    if variant.first_year is not None:
        keep = (years >= variant.first_year).to_numpy()

    kept_years = years[keep].reset_index(drop=True)
    train = training_mask(kept_years, config)
    weights = recency_weights(
        kept_years.to_numpy("int64"), config.trentino.test_years_start - 1, variant.half_life
    )
    return Design(
        features=features.loc[keep, columns].reset_index(drop=True),
        labels=labels[keep],
        years=kept_years,
        train=train,
        weights=weights,
    )


def experiment_directory(config: Config, name: str) -> Path:
    return config.path(config.paths.trentino_model_dir) / EXPERIMENTS_DIR / name


def reference_metrics(config: Config) -> dict[str, Any]:
    path = config.path(config.paths.trentino_model_dir) / REFERENCE_VERSION / METRICS_FILENAME
    metrics: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def fit_variant(
    config: Config, variant: Variant, data: Design, tuning: dict[str, Any]
) -> tuple[Estimator, dict[str, Any]]:
    """Year-blocked CV and one fit on the training years, both at v2's hyperparameters."""
    spec = SPECS["xgboost"]
    train = data.train
    train_weights = None if data.weights is None else data.weights[train]
    out_of_fold, _, folds = cross_validate(
        spec,
        data.features.loc[train],
        data.labels[train],
        data.years.loc[train],
        config,
        tuning["params"],
        train_weights,
    )
    estimator, columns, holdout = fit_holdout(
        spec, data.features, data.labels, train, config, tuning, data.weights
    )
    held_labels = data.labels[~train]
    metrics: dict[str, Any] = {
        "name": variant.name,
        "question": variant.question,
        "half_life": variant.half_life,
        "first_year": variant.first_year,
        "train_years": [int(data.years[train].min()), int(data.years[train].max())],
        "rows": {"train": int(train.sum()), "holdout": int((~train).sum())},
        "columns": columns,
        "folds": folds,
        "pooled_out_of_fold": scores(data.labels[train], out_of_fold),
        "holdout": scores(held_labels, holdout),
        "holdout_intervals": bootstrap_scores(
            held_labels,
            holdout,
            config.evaluation.bootstrap_resamples,
            config.project.random_seed,
        ),
    }
    logger.info(
        "%-14s | pooled AUPRC %.4f | holdout AUPRC %.4f AUROC %.4f",
        variant.name,
        metrics["pooled_out_of_fold"]["auprc"],
        metrics["holdout"]["auprc"],
        metrics["holdout"]["auroc"],
    )
    return estimator, metrics


def load_or_fit(
    config: Config,
    set_name: str,
    variant: Variant,
    table: tuple[pd.DataFrame, npt.NDArray[np.int8], pd.Series],
    reference: dict[str, Any],
    force: bool = False,
) -> tuple[Estimator, dict[str, Any]]:
    """A variant's fit from its directory, or a new one written there."""
    from xgboost import XGBClassifier

    directory = experiment_directory(config, set_name) / variant.name
    model_path = directory / HOLDOUT_MODEL_FILENAME
    metrics_path = directory / VARIANT_FILENAME
    if model_path.is_file() and metrics_path.is_file() and not force:
        estimator: Estimator = XGBClassifier()
        cast(Any, estimator).load_model(model_path)
        stored: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        logger.info("%s: reusing %s", variant.name, directory)
        return estimator, stored

    features, labels, years = table
    data = design(variant, features, labels, years, reference["columns"], config)
    directory.mkdir(parents=True, exist_ok=True)
    tuning = retune(config, variant, data, directory, force) if variant.retunes else None
    estimator, metrics = fit_variant(config, variant, data, tuning or reference["tuning"])
    if tuning is not None:
        metrics["tuning"] = {key: value for key, value in tuning.items() if key != "trial_table"}
    cast(BoostedEstimator, estimator).save_model(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return estimator, metrics


def retune(
    config: Config, variant: Variant, data: Design, directory: Path, force: bool = False
) -> dict[str, Any]:
    """The variant's own Optuna search on its training years, kept beside its fit."""
    from tfire.models.trentino import tune

    path = directory / TUNING_FILENAME
    if path.is_file() and not force:
        logger.info("%s: reusing the search in %s", variant.name, path)
        stored: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return stored

    settings = config.trentino.model_copy(
        update={
            key: value
            for key, value in (
                ("tuning_objective", variant.objective),
                ("search_space", variant.search_space),
            )
            if value is not None
        }
    )
    tuned_config = config.model_copy(update={"trentino": settings})
    train = data.train
    logger.info(
        "%s: searching on %s inside the %s bounds",
        variant.name,
        settings.tuning_objective,
        settings.search_space,
    )
    tuning = tune(data.features.loc[train], data.labels[train], data.years.loc[train], tuned_config)
    path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    if tuning["boundary"]:
        logger.warning("%s: the search ended on the edge of %s", variant.name, tuning["boundary"])
    return tuning


@dataclass(frozen=True)
class Bundle:
    """What scoring needs from a fitted variant."""

    estimator: Estimator
    columns: list[str]


def select(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The full design matrix cut to a model's own columns, in its order."""
    wanted = set(columns)
    return align_columns(features[[c for c in features.columns if c in wanted]], columns)


def raw_scores(bundle: Bundle, features: pd.DataFrame) -> FloatArray:
    return np.asarray(
        bundle.estimator.predict_proba(select(features, bundle.columns))[:, 1], dtype="float64"
    )


def logit(probabilities: FloatArray) -> FloatArray:
    clipped = np.clip(probabilities, _LOGIT_CLIP, 1 - _LOGIT_CLIP)
    return np.asarray(np.log(clipped) - np.log1p(-clipped), dtype="float64")


@dataclass
class CellShare:
    """Running share of variance explained by the cell, over days added one at a time.

    The same quantity as `events.cell_effect_share` without holding every day in memory.
    """

    cells: npt.NDArray[np.int64] | None = None
    sums: FloatArray | None = None
    total: float = 0.0
    total_squares: float = 0.0
    days: int = 0

    def add(self, cells: npt.NDArray[np.int64], values: FloatArray) -> None:
        if self.cells is None or self.sums is None:
            self.cells = cells.copy()
            self.sums = np.zeros(len(values))
        elif not np.array_equal(cells, self.cells):
            raise ValueError("Days scored on different cell sets cannot share a cell effect")
        self.sums += values
        self.total += float(values.sum())
        self.total_squares += float(np.square(values).sum())
        self.days += 1

    def share(self) -> float:
        if self.sums is None or not self.days:
            raise ValueError("No day added")
        count = self.days * len(self.sums)
        mean = self.total / count
        total_variance = self.total_squares / count - mean**2
        if total_variance <= 0:
            return 0.0
        between = float(np.square(self.sums / self.days).mean()) - mean**2
        return between / total_variance


@dataclass
class AxisTally:
    """Everything one variant accumulates over the event-axis days."""

    ranks: list[pd.DataFrame] = field(default_factory=list)
    shares: dict[str, CellShare] = field(default_factory=dict)
    mass: FloatArray | None = None


def axis_days(config: Config) -> dict[str, list[date]]:
    from tfire.models.danger import reference_days
    from tfire.models.events import VARIANCE_STRIDE, ignition_events, season_window

    events = ignition_events(config)
    first = date(config.trentino.test_years_start, 1, 1)
    span = (config.date_range.end - first).days + 1
    return {
        "events": sorted(set(pd.DatetimeIndex(events["date"]).date)),
        "decade": reference_days(config)[::VARIANCE_STRIDE],
        "august": season_window(config, 8),
        "april": season_window(config, 4),
        "mass": [first + timedelta(days=k) for k in range(0, span, MASS_STRIDE_DAYS)],
    }


def event_axis(
    config: Config, bundles: dict[str, Bundle], with_roads: bool = True
) -> dict[str, Any]:
    """Score every axis day once and read each variant off the same assembled frames."""
    from tfire.inference import GridScorer, within_day_rank
    from tfire.models.events import aggregate_percentiles, history_baseline, ignition_events

    events = ignition_events(config)
    days = axis_days(config)
    wanted = sorted(set().union(*(set(group) for group in days.values())))
    groups = {name: set(group) for name, group in days.items()}
    event_cells = {
        day: events.loc[pd.DatetimeIndex(events["date"]).date == day, "cell_id"].to_numpy()
        for day in groups["events"]
    }

    scorer = GridScorer(config, wanted, holdout=True)
    stored = Bundle(scorer.estimator, scorer.columns)
    registry = load_registry()
    tallies = {name: AxisTally() for name in bundles}
    logger.info("Event axis: %d day(s) for %d variant(s)", len(wanted), len(bundles))

    cells: npt.NDArray[np.int64] | None = None
    for index, day in enumerate(wanted):
        frame = scorer.frame(day)
        features, _, _ = design_matrix(frame.assign(is_fire=False), registry)
        day_cells = frame["cell_id"].to_numpy("int64")
        if cells is None:
            cells = day_cells
        if index == 0 and REFERENCE_NAME in bundles:
            check_reference(
                raw_scores(bundles[REFERENCE_NAME], features), raw_scores(stored, features)
            )

        for name, bundle in bundles.items():
            raw = raw_scores(bundle, features)
            tally = tallies[name]
            if day in groups["events"]:
                rank = pd.Series(within_day_rank(raw), index=pd.Index(day_cells, name="cell_id"))
                found = rank.reindex(event_cells[day])
                if found.isna().any():
                    raise ValueError(f"An ignition cell on {day} is not on the scored grid")
                tally.ranks.append(
                    pd.DataFrame(
                        {
                            "cell_id": event_cells[day],
                            "date": pd.Timestamp(day),
                            "within_day_percentile": found.to_numpy(),
                        }
                    )
                )
            margin = logit(raw)
            for group in ("decade", "august", "april"):
                if day in groups[group]:
                    tally.shares.setdefault(group, CellShare()).add(day_cells, margin)
            if day in groups["mass"]:
                if not np.array_equal(day_cells, cells):
                    raise ValueError("The scored grid changed between days")
                odds = raw / (1 - np.clip(raw, None, 1 - _LOGIT_CLIP))
                tally.mass = odds if tally.mass is None else tally.mass + odds
        if index and index % 100 == 0:
            logger.info("  axis: %d/%d days", index, len(wanted))

    if cells is None:
        raise ValueError("No day scored")
    baseline = history_baseline(config, config.trentino.test_years_start - 1)
    events_scored = events[["cell_id", "date", "season"]]
    baseline_frame = events_scored.assign(
        within_day_percentile=baseline.reindex(events_scored["cell_id"]).to_numpy()
    )
    roads = road_rows(config, cells, events) if with_roads else {}

    results: dict[str, Any] = {
        "days": {name: len(group) for name, group in days.items()},
        "baseline": summarize_events(baseline_frame, aggregate_percentiles),
        "events": {
            "cell_id": events_scored["cell_id"].astype("int64").tolist(),
            "date": pd.DatetimeIndex(events_scored["date"]).strftime("%Y-%m-%d").tolist(),
            "season": events_scored["season"].astype(str).tolist(),
        },
        "variants": {},
    }
    for name, tally in tallies.items():
        scored = events_scored.merge(
            pd.concat(tally.ranks), on=["cell_id", "date"], how="left", validate="m:1"
        )
        results["variants"][name] = {
            "events": summarize_events(scored, aggregate_percentiles),
            "percentiles": scored["within_day_percentile"].astype("float64").tolist(),
            "cell_share": {group: share.share() for group, share in tally.shares.items()},
            "roads": {
                road: observed_expected(corridor, tally.mass, observed)
                for road, (corridor, observed) in roads.items()
            }
            if tally.mass is not None
            else {},
        }
    results["roads_total"] = int(len(events))
    return results


def check_reference(refit: FloatArray, stored: FloatArray) -> None:
    """The refitted reference has to score a day as the stored holdout fit does."""
    gap = float(np.max(np.abs(refit - stored)))
    if gap > 1e-6:
        raise ValueError(f"The refitted v2 differs from the stored holdout fit by {gap:.2e}")
    logger.info("Refitted v2 matches the stored holdout fit (max gap %.1e)", gap)


def summarize_events(
    frame: pd.DataFrame, aggregate: Callable[[pd.DataFrame], dict[str, Any]]
) -> dict[str, Any]:
    return {
        "overall": aggregate(frame),
        "by_season": {
            str(season): aggregate(part) for season, part in frame.groupby("season", observed=True)
        },
    }


def road_rows(
    config: Config, cells: npt.NDArray[np.int64], events: pd.DataFrame
) -> dict[str, tuple[BoolArray, int]]:
    """Per road, its corridor over the scored cells and the holdout ignitions inside it."""
    from tfire.grid import load_grid
    from tfire.models.roads import corridors

    _, grid = load_grid(config)
    xy = grid.set_index("cell_id").loc[cells, ["x_coordinate", "y_coordinate"]]
    by_road = corridors(config, xy.to_numpy("float64"))
    inside = pd.Index(cells)
    rows = {}
    for road, corridor in by_road.items():
        members = set(inside[corridor])
        rows[road] = (corridor, int(events["cell_id"].isin(members).sum()))
    return rows


def observed_expected(corridor: BoolArray, mass: FloatArray, observed: int) -> dict[str, float]:
    """Holdout ignitions in a corridor, and the share of the variant's mass that falls there.

    The mass is the summed odds, which for rare events is proportional to the population rate
    whatever the case-control offset; the total is pinned to the ignitions actually recorded,
    since the question is where a variant puts its fires, not how many.
    """
    return {"observed": float(observed), "share": float(mass[corridor].sum() / mass.sum())}


def finish_roads(results: dict[str, Any]) -> None:
    """Turn each variant's corridor share into expected ignitions and a Poisson p-value."""
    from tfire.models.roads import poisson_p

    total = results["roads_total"]
    for variant in results["variants"].values():
        for road in variant["roads"].values():
            road["expected"] = road["share"] * total
            road["p"] = poisson_p(int(road["observed"]), road["expected"])


def paired_median_gap(
    reference: FloatArray,
    other: FloatArray,
    days: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Median percentile of `other` minus the reference's, with a bootstrap over ignition days.

    Both variants are read on the same resample, so the interval is for the gap and not for
    either median. Whole days are drawn, since the ignitions of one day share one map.
    """
    codes, uniques = pd.factorize(pd.Index(days))
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    rng = np.random.default_rng(seed)
    gaps = np.empty(resamples)
    for b in range(resamples):
        rows = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        gaps[b] = np.median(other[rows]) - np.median(reference[rows])
    low, high = np.quantile(gaps, [0.025, 0.975])
    return {
        "delta": float(np.median(other) - np.median(reference)),
        "lo": float(low),
        "hi": float(high),
    }


def finish_gaps(results: dict[str, Any], config: Config) -> None:
    """Each variant's median gap to the reference, over all ignitions and per season."""
    events = results["events"]
    seasons = np.asarray(events["season"])
    dates = np.asarray(events["date"])
    variants = results["variants"]
    base = np.asarray(variants[REFERENCE_NAME]["percentiles"])
    subsets = {"overall": np.ones(len(seasons), dtype=bool)}
    subsets |= {str(season): seasons == season for season in sorted(set(seasons))}
    for variant in variants.values():
        other = np.asarray(variant["percentiles"])
        variant["median_gap"] = {
            name: paired_median_gap(
                base[rows],
                other[rows],
                list(dates[rows]),
                config.evaluation.bootstrap_resamples,
                config.project.random_seed,
            )
            for name, rows in subsets.items()
        }


def run_set(config: Config, name: str, force: bool = False) -> dict[str, Any]:
    """Fit every variant of a set, read them all on the event axis, write the results."""
    from tfire.models.evaluate import check_reproducible

    if name not in VARIANT_SETS:
        raise ValueError(f"Unknown variant set {name!r}; known: {sorted(VARIANT_SETS)}")
    reference = reference_metrics(config)
    frame = pd.read_parquet(config.path(config.paths.dataset_out))
    table = design_matrix(frame, load_registry())

    bundles: dict[str, Bundle] = {}
    fitted: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_SETS[name]:
        estimator, metrics = load_or_fit(config, name, variant, table, reference, force)
        if variant.name == REFERENCE_NAME:
            check_reproducible(reference["models"]["xgboost"]["holdout"], metrics["holdout"])
        bundles[variant.name] = Bundle(estimator, list(metrics["columns"]))
        fitted[variant.name] = metrics

    axis = event_axis(config, bundles)
    finish_roads(axis)
    finish_gaps(axis, config)
    results = {
        "set": name,
        "fits": fitted,
        "axis": axis,
        "reference_tuning": reference_search(reference),
    }
    directory = experiment_directory(config, name)
    (directory / RESULTS_FILENAME).write_text(json.dumps(results, indent=2), encoding="utf-8")

    out = config.path(config.paths.report_dir) / f"variants_{name}.md"
    out.write_text(render(results), encoding="utf-8")
    logger.info("Wrote %s", out)
    return results


def reference_search(reference: dict[str, Any]) -> dict[str, Any]:
    """v2's tuned settings, with where they sit against the bounds v2 was searched in."""
    from tfire.models.trentino import SEARCH_SPACES, on_the_boundary

    params = dict(reference["tuning"]["params"])
    return {
        "params": params,
        "rounds": reference["tuning"]["rounds"],
        "boundary": on_the_boundary(params, SEARCH_SPACES["v2"]),
    }


def _interval(metrics: dict[str, Any]) -> str:
    low = metrics["holdout_intervals"]["auprc"]["lo"]
    high = metrics["holdout_intervals"]["auprc"]["hi"]
    return f"{metrics['holdout']['auprc']:.4f} ({low:.3f}-{high:.3f})"


def criteria(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every variant's gaps to the refitted v2, and whether it passes."""
    fits, axis = results["fits"], results["axis"]["variants"]
    base_fit, base_axis = fits[REFERENCE_NAME], axis[REFERENCE_NAME]
    out = {}
    for name in fits:
        auprc = fits[name]["holdout"]["auprc"] - base_fit["holdout"]["auprc"]
        events = axis[name]["events"]
        median = (
            events["overall"]["median_percentile"]
            - base_axis["events"]["overall"]["median_percentile"]
        )
        summer = events["by_season"].get("summer", {}).get("median_percentile", float("nan"))
        base_summer = base_axis["events"]["by_season"].get("summer", {})
        gaps = axis[name].get("median_gap", {})
        out[name] = {
            "auprc_delta": auprc,
            "auprc_ok": auprc >= -AUPRC_TOLERANCE,
            "median_delta": median,
            "median_ok": median >= 0,
            "median_gap": gaps.get("overall"),
            "summer_delta": summer - base_summer.get("median_percentile", float("nan")),
            "summer_gap": gaps.get("summer"),
            "august_delta": axis[name]["cell_share"]["august"] - base_axis["cell_share"]["august"],
        }
    return out


def _gap(delta: float, gap: dict[str, float] | None) -> str:
    if gap is None:
        return f"{delta:+.3f}"
    return f"{delta:+.3f} ({gap['lo']:+.3f} to {gap['hi']:+.3f})"


def render(results: dict[str, Any]) -> str:
    from tfire.report import table

    fits, axis = results["fits"], results["axis"]
    names = list(fits)
    seasons = ["winter", "spring", "summer", "autumn"]
    judged = criteria(results)
    lines = [
        f"# Retraining variants: {results['set'].replace('_', ' ')}",
        "",
        "Generated by `tfire variants`. Every variant keeps v2's tuned hyperparameters and is "
        "fitted on the training years only; the holdout years are never dropped, so every row "
        "is scored on the same 2015-2024. The event axis reads all variants off the same "
        f"assembled days ({', '.join(f'{k} {v}' for k, v in axis['days'].items())}).",
        "",
        "## The variants",
        "",
    ]
    lines += table(
        ["variant", "question", "training years", "rows", "columns"],
        [
            [
                f"`{name}`",
                fits[name]["question"],
                "-".join(str(year) for year in fits[name]["train_years"]),
                fits[name]["rows"]["train"],
                len(fits[name]["columns"]),
            ]
            for name in names
        ],
    )
    searched = [name for name in names if "tuning" in fits[name]]
    if searched:
        reference = results.get("reference_tuning", {})
        rows: list[tuple[str, dict[str, Any], Any, str, list[str]]] = [
            (
                "v2",
                reference.get("params", {}),
                reference.get("rounds"),
                "pooled_auprc",
                reference.get("boundary", []),
            )
        ]
        rows += [
            (
                name,
                fits[name]["tuning"]["params"],
                fits[name]["tuning"]["rounds"],
                fits[name]["tuning"]["objective"],
                fits[name]["tuning"].get("boundary", []),
            )
            for name in searched
        ]
        keys = sorted({key for _, params, *_ in rows for key in params})
        lines += [
            "",
            "## The search",
            "",
            "Best trial of each search, on the training years only. A parameter listed under "
            "edge ended within 2% of the span from a bound.",
            "",
        ]
        lines += table(
            ["variant", "objective", "rounds", *keys, "edge"],
            [
                [
                    f"`{name}`",
                    objective,
                    rounds,
                    *(f"{params[key]:.3g}" if key in params else "n/a" for key in keys),
                    ", ".join(edge) or "none",
                ]
                for name, params, rounds, objective, edge in rows
            ],
        )
    lines += ["", "## Holdout 2015-2024", ""]
    lines += table(
        ["variant", "AUPRC (95% CI)", "vs v2", "AUROC", "lift", "pooled out-of-fold AUPRC"],
        [
            [
                f"`{name}`",
                _interval(fits[name]),
                f"{judged[name]['auprc_delta']:+.4f}",
                f"{fits[name]['holdout']['auroc']:.4f}",
                f"{fits[name]['holdout']['lift']:.2f}",
                f"{fits[name]['pooled_out_of_fold']['auprc']:.4f}",
            ]
            for name in names
        ],
    )
    lines += [
        "",
        "## Event axis: where the holdout ignitions land in their own day's map",
        "",
        "Within-day percentile on the estimator's score, median over the ignitions. The fire-"
        "history baseline is the kernel density of the training-years cadastre.",
        "",
    ]

    def event_row(label: str, events: dict[str, Any]) -> list[str]:
        overall = events["overall"]
        return [
            label,
            f"{overall['median_percentile']:.3f}",
            f"{100 * overall['share_at_or_above_90']:.1f}%",
            *(
                f"{events['by_season'][season]['median_percentile']:.3f}"
                if season in events["by_season"]
                else "n/a"
                for season in seasons
            ),
        ]

    lines += table(
        ["variant", "median", "≥ 90th", *(f"{season} median" for season in seasons)],
        [event_row(f"`{name}`", axis["variants"][name]["events"]) for name in names]
        + [event_row("fire-history baseline", axis["baseline"])],
    )
    lines += [
        "",
        "## How much of the map is the cell",
        "",
        "Share of the variance in the score's logit explained by which cell a value belongs to, "
        "over one holdout day in ten (decade) and over 15 consecutive days of August and of April "
        "2024. Lower means the weather moves the map more.",
        "",
    ]
    lines += table(
        ["variant", "decade", "August", "April"],
        [
            [
                f"`{name}`",
                *(
                    f"{100 * axis['variants'][name]['cell_share'][group]:.1f}%"
                    for group in ("decade", "august", "april")
                ),
            ]
            for name in names
        ],
    )
    roads = list(axis["variants"][names[0]]["roads"])
    if roads:
        lines += [
            "",
            "## Main roads, observed against expected 2015-2024",
            "",
            "Ignitions within 1 km of the road against the share of the variant's summed odds "
            f"that falls there, times the {axis['roads_total']} holdout ignitions. p is the "
            "two-sided Poisson test.",
            "",
        ]
        lines += table(
            ["variant", *roads],
            [
                [
                    f"`{name}`",
                    *(
                        f"{int(cell['observed'])} / {cell['expected']:.1f} (p {cell['p']:.2f})"
                        for cell in (axis["variants"][name]["roads"][road] for road in roads)
                    ),
                ]
                for name in names
            ],
        )
    lines += [
        "",
        "## Criteria",
        "",
        f"AUPRC no more than {AUPRC_TOLERANCE} below v2; median percentile of the holdout "
        "ignitions not lower than v2's; summer and the August cell share reported beside them. "
        "The intervals on the medians are a paired bootstrap over ignition days: both variants "
        "are read on the same resampled days, so an interval that spans zero means the gap is "
        "not distinguishable from v2.",
        "",
    ]
    lines += table(
        ["variant", "AUPRC", "median", "summer median", "August cell share", "passes"],
        [
            [
                f"`{name}`",
                f"{row['auprc_delta']:+.4f}",
                _gap(row["median_delta"], row["median_gap"]),
                _gap(row["summer_delta"], row["summer_gap"]),
                f"{100 * row['august_delta']:+.1f} pt",
                "yes" if row["auprc_ok"] and row["median_ok"] else "no",
            ]
            for name, row in judged.items()
            if name != REFERENCE_NAME
        ],
    )
    lines.append("")
    return "\n".join(lines)
