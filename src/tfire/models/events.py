"""Where recorded ignitions landed in the map drawn for their own day."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.features.human import calendar_features
from tfire.models.danger import CLASS_KEYS, DangerClasses, reference_days

logger = logging.getLogger(__name__)

EVENTS_FILENAME: Final = "events.json"

# thresholds the shares are reported at, on the within-day percentile
_SHARE_AT: Final = (0.90, 0.99)

# the variance diagnostic scores one holdout day in this many, which keeps it around a year of
# days: enough to separate a fixed cell effect from a day effect, cheap enough to run per variant
_VARIANCE_STRIDE: Final = 10

# the same diagnostic over a run of consecutive days in one fire season. The two answer different
# questions and give very different numbers: spread over a decade the weather moves a great deal
# and the cell effect looks small, while inside one August almost the only thing that separates
# two cells is what they are, which is exactly the window an operator compares maps in.
_WINDOW_DAYS: Final = 15
_WINDOW_MONTH: Final = 8


def ignition_events(config: Config) -> pd.DataFrame:
    """Recorded ignitions in the holdout years, on the cells and dates the labels used."""
    samples = pd.read_parquet(config.path(config.paths.samples_out))
    fires = pd.read_parquet(
        config.path(config.paths.fires_out), columns=["fire_id", "area_ha", "cause"]
    )

    events = samples.loc[samples["is_fire"], ["cell_id", "date", "fire_id"]].copy()
    events = events[events["date"].dt.year >= config.trentino.test_years_start]
    events = events.merge(fires, on="fire_id", how="left", validate="m:1")

    calendar = calendar_features(pd.DatetimeIndex(events["date"].unique()))
    events = events.merge(calendar[["date", "season", "month"]], on="date", how="left")
    ordered: pd.DataFrame = events.sort_values(["date", "cell_id"]).reset_index(drop=True)
    return ordered


def _shares(percentiles: npt.NDArray[np.float64]) -> dict[str, float]:
    return {
        f"share_at_or_above_{int(round(threshold * 100))}": float((percentiles >= threshold).mean())
        for threshold in _SHARE_AT
    }


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    percentiles = frame["within_day_percentile"].to_numpy(dtype="float64")
    return {
        "events": int(len(frame)),
        "median_percentile": float(np.median(percentiles)),
        "mean_percentile": float(percentiles.mean()),
        **_shares(percentiles),
    }


def history_baseline(config: Config, last_year: int) -> pd.Series:
    """Within-grid percentile of the cadastre's ignition density, `start` to `last_year`."""
    from tfire.features.history import kernel_density
    from tfire.grid import load_grid
    from tfire.sampling import prepare_fires

    spec, grid = load_grid(config)
    fires = prepare_fires(pd.read_parquet(config.path(config.paths.fires_out)), spec)
    year = fires["ignition_date"].dt.year
    fires = fires[
        (fires["cell_id"] >= 0) & (year >= config.date_range.start.year) & (year <= last_year)
    ]
    density = kernel_density(
        spec,
        fires["x"].to_numpy(dtype="float64"),
        fires["y"].to_numpy(dtype="float64"),
        config.history.bandwidth_m,
    )
    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy("int64")
    percentile = pd.Series(density[active]).rank(pct=True).to_numpy("float64")
    logger.info(
        "Fire-history baseline: %d ignition(s), %d to %d",
        len(fires),
        config.date_range.start.year,
        last_year,
    )
    return pd.Series(percentile, index=pd.Index(active, name="cell_id"), name="baseline")


def own_classes(config: Config, scored: dict[date, npt.NDArray[np.float64]]) -> DangerClasses:
    """Breaks cut on the scoring model's own probabilities over a sample of holdout days."""
    values = np.concatenate([scored[day] for day in sorted(scored)])
    percentiles = list(config.trentino.danger_percentiles)
    return DangerClasses(
        breaks=[float(value) for value in np.percentile(values, percentiles)],
        percentiles=percentiles,
        class_keys=list(CLASS_KEYS[: len(percentiles) + 1]),
        reference=f"holdout fit, full grid, {len(scored)} sampled holdout day(s)",
        reference_years=[config.trentino.test_years_start, config.date_range.end.year],
        rows=int(values.size),
        model_version=config.trentino.version,
        config_sha256=config.digest(),
    )


def cell_effect_share(scored: dict[date, npt.NDArray[np.float64]]) -> float:
    """Fraction of the variance in log10(p) explained by which cell a value belongs to.

    One near this ceiling says the map is a fixed picture of the province that the weather
    only tints, which is what the whole ranking is meant not to be.
    """
    matrix = np.log10(np.vstack([scored[day] for day in sorted(scored)]))
    total = float(matrix.var())
    if total == 0.0:
        return 0.0
    return float(matrix.mean(axis=0).var() / total)


def verify_events(config: Config, classes: DangerClasses | None = None) -> dict[str, Any]:
    """Score every day that carries a holdout ignition and report where the ignitions landed."""
    from tfire.inference import GridScorer

    events = ignition_events(config)
    if events.empty:
        raise ValueError(
            f"No recorded ignition at or after {config.trentino.test_years_start}, "
            "so there is nothing to verify against."
        )

    stamps = pd.DatetimeIndex(events["date"])
    event_days = set(stamps.date)
    variance_days = set(reference_days(config)[::_VARIANCE_STRIDE])
    window_days = set(_season_window(config))
    wanted = sorted(event_days | variance_days | window_days)

    scorer = GridScorer(config, wanted, holdout=True)
    logger.info(
        "Verifying %d ignition(s) over %d day(s), plus %d day(s) for the variance split",
        len(events),
        len(event_days),
        len(variance_days),
    )

    per_day = stamps.date
    cells_by_day = {
        day: events.loc[per_day == day, "cell_id"].to_numpy() for day in sorted(event_days)
    }

    picked: list[pd.DataFrame] = []
    variance_scores: dict[date, npt.NDArray[np.float64]] = {}
    window_scores: dict[date, npt.NDArray[np.float64]] = {}
    days_scored = 0
    for index, day in enumerate(wanted):
        frame, _, probability, rank = scorer.day(day)
        if day in variance_days:
            variance_scores[day] = probability.astype("float64")
        if day in window_days:
            window_scores[day] = probability.astype("float64")
        if day in event_days:
            days_scored += 1
            table = pd.DataFrame(
                {"probability": probability, "within_day_percentile": rank},
                index=pd.Index(frame["cell_id"].to_numpy(), name="cell_id"),
            )
            wanted_cells = cells_by_day[day]
            found = table.reindex(wanted_cells)
            if found["probability"].isna().any():
                missing = sorted(set(wanted_cells) - set(table.index))
                raise ValueError(f"Cell(s) {missing} are not on the grid scored for {day}")
            picked.append(found.assign(date=pd.Timestamp(day)).reset_index())
        if index and index % 50 == 0:
            logger.info("  %d/%d days", index, len(wanted))

    scored = events.merge(pd.concat(picked), on=["cell_id", "date"], how="left", validate="m:1")
    classes = classes or own_classes(config, variance_scores)
    scored["danger_class"] = classes.classify(scored["probability"].to_numpy())

    last_training_year = config.trentino.test_years_start - 1
    baseline = scored[["season"]].assign(
        within_day_percentile=history_baseline(config, last_training_year)
        .reindex(scored["cell_id"])
        .to_numpy()
    )

    counts = scored["danger_class"].value_counts()
    return {
        "reference": "recorded ignitions, PAT cadastre",
        "model": f"fit on {config.date_range.start.year}-{last_training_year} only",
        "years": [config.trentino.test_years_start, config.date_range.end.year],
        "days_scored": days_scored,
        "overall": _aggregate(scored),
        "by_season": {
            str(season): _aggregate(part)
            for season, part in scored.groupby("season", observed=True)
        },
        "baseline": {
            "name": "fire-history density, "
            f"{config.history.bandwidth_m / 1000:g} km kernel, "
            f"{config.date_range.start.year}-{last_training_year}",
            "overall": _aggregate(baseline),
            "by_season": {
                str(season): _aggregate(part)
                for season, part in baseline.groupby("season", observed=True)
            },
        },
        "class_breaks": {"reference": classes.reference, "breaks": classes.breaks},
        "class_distribution": {
            key: int(counts.get(index, 0)) for index, key in enumerate(classes.class_keys)
        },
        "cell_effect_variance_share": cell_effect_share(variance_scores),
        "cell_effect_variance_share_in_season": cell_effect_share(window_scores),
        "variance_days": [day.isoformat() for day in sorted(variance_scores)],
        "season_window": [day.isoformat() for day in sorted(window_scores)],
    }


def _season_window(config: Config) -> list[date]:
    """A run of consecutive days in the last holdout August, for the within-season split."""
    year = config.date_range.end.year
    first = date(year, _WINDOW_MONTH, 1)
    return [first + timedelta(days=offset) for offset in range(_WINDOW_DAYS)]
