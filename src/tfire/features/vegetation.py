"""Monthly Landsat vegetation indices, surface temperature and observation quality per cell.

`ndwi` is the NIR/SWIR form (Gao), which tracks canopy water content, not the green/NIR form
that maps open water.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import GridSpec, load_grid
from tfire.raster import read_cell_bands
from tfire.sources.landsat import (
    INDEX_NAMES,
    OUTPUT_BANDS,
    cached_months,
    month_starts,
    parse_band_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

VALUE_NAMES: Final = (*INDEX_NAMES, "lst")
QUALITY_NAMES: Final = ("valid_fraction", "snow_fraction", "age_days")
FEATURE_NAMES: Final = (*VALUE_NAMES, *QUALITY_NAMES)


def _months_and_bands(
    paths: Iterable[Path], spec: GridSpec, active: npt.NDArray[np.int64]
) -> tuple[list[date], dict[str, npt.NDArray[np.float32]]]:
    """Read the cached rasters into one array per band, shaped (month, active cell)."""
    per_month: dict[date, dict[str, npt.NDArray[np.float32]]] = {}
    for path in paths:
        for description, values in read_cell_bands(spec, path).items():
            name, month = parse_band_name(description)
            per_month.setdefault(month, {})[name] = values[active].astype("float32")

    months = sorted(per_month)
    stacked = {
        name: np.stack([per_month[month][name] for month in months]) for name in OUTPUT_BANDS
    }
    return months, stacked


def apply_plausibility(
    values: dict[str, npt.NDArray[np.float32]], config: Config
) -> dict[str, npt.NDArray[np.float32]]:
    """Drop values a correct pipeline cannot produce.

    A normalized difference outside [-1, 1] means a scaling error; EVI leaves the same range
    when its denominator approaches zero over bright or snowy ground.
    """
    index_bounds = (config.vegetation.min_index, config.vegetation.max_index)
    bounds = {name: index_bounds for name in INDEX_NAMES}
    bounds["lst"] = (config.vegetation.min_lst_c, config.vegetation.max_lst_c)

    cleaned = dict(values)
    for name, (low, high) in bounds.items():
        band = cleaned[name].copy()
        outside = (band < low) | (band > high)
        dropped = int(np.count_nonzero(outside))
        if dropped:
            logger.info(
                "%s: %d cell-month(s) outside [%.1f, %.1f] dropped", name, dropped, low, high
            )
        band[outside] = np.nan
        cleaned[name] = band
    return cleaned


def forward_fill(
    values: dict[str, npt.NDArray[np.float32]],
    observed: npt.NDArray[np.bool_],
    months: list[date],
    max_fill_days: int,
) -> tuple[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
    """Carry each cell's last usable composite forward, and report how old it is."""
    ordinals = np.array([month.toordinal() for month in months], dtype="int64")
    positions = np.where(observed, np.arange(len(months), dtype="int64")[:, None], -1)
    source = np.maximum.accumulate(positions, axis=0)

    age = np.where(source >= 0, ordinals[:, None] - ordinals[np.maximum(source, 0)], -1)
    usable = (source >= 0) & (age <= max_fill_days)

    filled = {}
    for name, band in values.items():
        carried = np.take_along_axis(band, np.maximum(source, 0), axis=0)
        filled[name] = np.where(usable, carried, np.nan).astype("float32")
    return filled, np.where(usable, age, np.nan).astype("float32")


def build_vegetation_frame(config: Config) -> pd.DataFrame:
    spec, grid = load_grid(config)
    cached = cached_months(config)
    if not cached:
        raise FileNotFoundError(
            f"No Landsat rasters under {config.path(config.paths.landsat_raw)}. "
            "Run `tfire fetch-landsat` first."
        )

    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()
    months, bands = _months_and_bands(sorted(set(cached.values())), spec, active)

    values = apply_plausibility({name: bands[name] for name in VALUE_NAMES}, config)
    observed = bands["valid_fraction"] >= config.vegetation.min_valid_fraction
    values, age = forward_fill(values, observed, months, config.vegetation.max_fill_days)

    if config.vegetation.no_observation_strategy == "climatology":
        values = _fill_from_climatology(values, months, active, config)

    n_months, n_cells = age.shape
    frame = pd.DataFrame(
        {
            "cell_id": np.tile(active.astype("int32"), n_months),
            "date": np.repeat(np.array(months, dtype="datetime64[us]"), n_cells),
        }
    )
    for name in VALUE_NAMES:
        frame[name] = values[name].reshape(-1)
    frame["valid_fraction"] = bands["valid_fraction"].reshape(-1)
    frame["snow_fraction"] = bands["snow_fraction"].reshape(-1)
    frame["age_days"] = age.reshape(-1)
    return frame


def months_in_reach(config: Config, composite: date) -> list[date]:
    """The composite months that can still supply a value for `composite`."""
    return month_starts(composite - timedelta(days=config.vegetation.max_fill_days), composite)


def refresh_landsat(config: Config, day: date) -> list[Path]:
    """Fetch whatever composite a date needs and the cache does not hold."""
    from tfire.sources.landsat import fetch_months

    composite = preceding_month(day)
    wanted = months_in_reach(config, composite)
    missing = [month for month in wanted if month not in cached_months(config)]
    if not missing:
        logger.info("Every composite %s needs is already cached", day)
        return []

    logger.info(
        "%s needs %d month(s) not cached, %s to %s",
        day,
        len(missing),
        f"{missing[0]:%Y-%m}",
        f"{missing[-1]:%Y-%m}",
    )
    return fetch_months(config, missing)


def preceding_month(day: date) -> date:
    """The month before the one `day` falls in, which is the composite it may attach to."""
    first = date(day.year, day.month, 1)
    return date((first - timedelta(days=1)).year, (first - timedelta(days=1)).month, 1)


def _empty_vegetation(active: npt.NDArray[np.int64]) -> pd.DataFrame:
    frame = pd.DataFrame({"cell_id": active.astype("int32")})
    for name in FEATURE_NAMES:
        frame[name] = np.full(len(active), np.nan, dtype="float32")
    return frame


def operational_window(config: Config, composite: date) -> pd.DataFrame:
    """The vegetation block for one composite month, built from the trailing cached months.

    Carrying a value forward only ever reaches back `max_fill_days`, so the months inside that
    reach are the whole history this needs; nothing before them can change the answer. That is
    what lets a single date be served without rebuilding the 41-year table.
    """
    spec, grid = load_grid(config)
    reach = config.vegetation.max_fill_days
    wanted = months_in_reach(config, composite)

    cached = cached_months(config)
    available = [month for month in wanted if month in cached]
    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()

    if not available:
        logger.warning(
            "No Landsat composite cached within %d days of %s, so the vegetation block is "
            "empty for it. Run `tfire predict --refresh-vegetation` to fetch it.",
            reach,
            f"{composite:%Y-%m}",
        )
        return _empty_vegetation(active)

    paths = sorted({cached[month] for month in available})
    months, bands = _months_and_bands(paths, spec, active)

    # a chunk holds three months, so reading one drags in months outside the reach
    keep = [index for index, month in enumerate(months) if month in set(wanted)]
    months = [months[index] for index in keep]
    bands = {name: values[keep] for name, values in bands.items()}

    values = apply_plausibility({name: bands[name] for name in VALUE_NAMES}, config)
    observed = bands["valid_fraction"] >= config.vegetation.min_valid_fraction
    values, age = forward_fill(values, observed, months, reach)

    if months[-1] != composite:
        logger.warning(
            "No composite for %s; the newest inside reach is %s",
            f"{composite:%Y-%m}",
            f"{months[-1]:%Y-%m}",
        )

    frame = pd.DataFrame({"cell_id": active.astype("int32")})
    for name in VALUE_NAMES:
        frame[name] = values[name][-1]
    frame["valid_fraction"] = bands["valid_fraction"][-1]
    frame["snow_fraction"] = bands["snow_fraction"][-1]
    frame["age_days"] = age[-1]

    usable = float(frame["ndvi"].notna().mean())
    logger.info(
        "Vegetation for %s: %d month(s) in reach, %.1f%% of cells carry an ndvi, mean age %.0f d",
        f"{composite:%Y-%m}",
        len(months),
        100 * usable,
        float(np.nanmean(frame["age_days"])) if usable else float("nan"),
    )
    return frame


def build_climatology(vegetation: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-cell seasonal normals, from the months a cell was actually observed."""
    fresh = vegetation.loc[vegetation["age_days"] == 0].copy()
    fresh["month"] = fresh["date"].dt.month.astype("int16")

    statistics = ("mean", "std", "count")
    grouped = fresh.groupby(["cell_id", "month"], sort=True)[list(VALUE_NAMES)]
    climatology = grouped.agg(list(statistics))
    climatology.columns = [f"{name}_{stat}" for name in VALUE_NAMES for stat in statistics]
    climatology = climatology.reset_index()

    minimum = config.vegetation.climatology_min_years
    for name in VALUE_NAMES:
        thin = climatology[f"{name}_count"] < minimum
        climatology.loc[thin, [f"{name}_mean", f"{name}_std"]] = np.nan
        climatology[f"{name}_count"] = climatology[f"{name}_count"].astype("int16")
        for statistic in ("mean", "std"):
            climatology[f"{name}_{statistic}"] = climatology[f"{name}_{statistic}"].astype(
                "float32"
            )

    climatology["cell_id"] = climatology["cell_id"].astype("int32")
    return climatology


def _fill_from_climatology(
    values: dict[str, npt.NDArray[np.float32]],
    months: list[date],
    active: npt.NDArray[np.int64],
    config: Config,
) -> dict[str, npt.NDArray[np.float32]]:
    """Substitute the cell's seasonal normal wherever no observation is in reach."""
    path = config.path(config.paths.vegetation_climatology_out)
    if not path.exists():
        raise FileNotFoundError(
            f"no_observation_strategy is 'climatology' but {path} does not exist. "
            "Build the table once with the 'nan' strategy first."
        )

    climatology = pd.read_parquet(path).set_index(["cell_id", "month"])
    filled = dict(values)
    for name in VALUE_NAMES:
        normals = climatology[f"{name}_mean"]
        lookup = np.stack(
            [
                normals.reindex(pd.MultiIndex.from_product([active, [month.month]])).to_numpy()
                for month in months
            ]
        ).astype("float32")
        filled[name] = np.where(np.isnan(filled[name]), lookup, filled[name]).astype("float32")
    return filled


def preceding_composite(dates: pd.Series, composites: Iterable[date]) -> pd.Series:
    """The last composite whose month ends strictly before each date.

    The composite covering an ignition day also covers the days after it, and a burn scar drops
    NDVI and NBR hard, so using it would leak the label into the features. The same rule is what
    M12 carries vegetation forward with at the forecast end.
    """
    starts = np.array(sorted(composites), dtype="datetime64[M]")
    if not starts.size:
        raise ValueError("No composites to choose from")

    month = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[M]")
    position = np.searchsorted(starts, month, side="left") - 1
    chosen = starts[np.maximum(position, 0)].astype("datetime64[us]")
    return pd.Series(
        np.where(position >= 0, chosen, np.datetime64("NaT", "us")),
        index=dates.index,
        dtype="datetime64[us]",
    )


def missingness_report(vegetation: pd.DataFrame) -> pd.DataFrame:
    """Coverage per month of the record"""
    report = vegetation.assign(
        year=vegetation["date"].dt.year.astype("int16"),
        month=vegetation["date"].dt.month.astype("int16"),
    ).groupby(["year", "month"], sort=True)

    frame = pd.DataFrame(
        {
            "cells": report.size(),
            "valid_fraction_mean": report["valid_fraction"].mean(),
            "snow_fraction_mean": report["snow_fraction"].mean(),
            "observed_share": report["age_days"].apply(lambda age: float((age == 0).mean())),
            "unusable_share": report["age_days"].apply(lambda age: float(age.isna().mean())),
        }
    )
    for name in VALUE_NAMES:
        frame[f"{name}_missing_share"] = report[name].apply(lambda band: float(band.isna().mean()))
    return frame.reset_index()


def validate_vegetation(vegetation: pd.DataFrame, config: Config, months: int, cells: int) -> None:
    """Check the acceptance criteria, logging loudly."""
    expected = months * cells
    if len(vegetation) != expected:
        logger.error(
            "Expected %d rows (%d month(s) x %d cell(s)), got %d",
            expected,
            months,
            cells,
            len(vegetation),
        )

    low, high = config.vegetation.min_index, config.vegetation.max_index
    for name in INDEX_NAMES:
        outside = int((~vegetation[name].between(low, high) & vegetation[name].notna()).sum())
        if outside:
            logger.error("%s: %d value(s) outside [%.1f, %.1f]", name, outside, low, high)

    observed = float((vegetation["age_days"] == 0).mean())
    unusable = float(vegetation["age_days"].isna().mean())
    logger.info(
        "Coverage: %.1f%% of cell-months observed fresh, %.1f%% carried forward, "
        "%.1f%% with no observation in reach",
        100 * observed,
        100 * (1 - observed - unusable),
        100 * unusable,
    )


def extract_vegetation(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `vegetation.parquet`, one row per active cell and month."""
    out = config.path(config.paths.vegetation_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    record = month_starts(
        date(config.date_range.start.year, 1, 1), date(config.date_range.end.year, 12, 1)
    )
    cached = cached_months(config)
    missing = [month for month in record if month not in cached]
    if missing:
        logger.warning(
            "Building over a partial cache: %d of %d month(s) absent, first %s. "
            "Run `tfire fetch-landsat` to complete it.",
            len(missing),
            len(record),
            f"{missing[0]:%Y-%m}",
        )

    vegetation = build_vegetation_frame(config)
    validate_vegetation(vegetation, config, len(cached), vegetation["cell_id"].nunique())

    climatology = build_climatology(vegetation, config)
    report = missingness_report(vegetation)

    out.parent.mkdir(parents=True, exist_ok=True)
    vegetation.to_parquet(out, index=False)
    climatology.to_parquet(config.path(config.paths.vegetation_climatology_out), index=False)
    report.to_parquet(config.path(config.paths.vegetation_missingness_out), index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(vegetation), vegetation.shape[1], out)
    logger.info("Wrote %d climatology row(s) and %d report row(s)", len(climatology), len(report))
    return vegetation
