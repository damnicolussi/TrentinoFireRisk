"""Per-cell ignition density from the cadastre, over a window that stops before its own year."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import GridSpec, load_grid

logger = logging.getLogger(__name__)

DENSITY_COLUMN: Final = "ignition_density"

_M2_PER_KM2: Final = 1e6


def kernel_density(
    spec: GridSpec,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    bandwidth_m: float,
) -> npt.NDArray[np.float64]:
    """Gaussian kernel count per cell center, in events per km2."""
    density = np.zeros(spec.n_cells, dtype="float64")
    if x.size == 0:
        return density

    centers_x, centers_y = spec.cell_center(np.arange(spec.n_cells))
    # a Gaussian is negligible past four bandwidths, and the cutoff turns an all-pairs
    # product over 24,842 cells x 3,187 events into a local one
    reach = 4.0 * bandwidth_m
    scale = 1.0 / (2.0 * np.pi * bandwidth_m**2) * _M2_PER_KM2

    for event_x, event_y in zip(x, y, strict=True):
        near = (np.abs(centers_x - event_x) <= reach) & (np.abs(centers_y - event_y) <= reach)
        if not near.any():
            continue
        squared = (centers_x[near] - event_x) ** 2 + (centers_y[near] - event_y) ** 2
        density[near] += scale * np.exp(-squared / (2.0 * bandwidth_m**2))

    return density


def extract_history(config: Config, force: bool = False) -> pd.DataFrame:
    """One ignition-density field per year, built only from the years strictly before it."""
    out = config.path(config.paths.fire_history_out)
    if out.is_file() and not force:
        logger.info("%s already exists, use --force to rebuild", out)
        return pd.read_parquet(out)

    from tfire.sampling import prepare_fires

    spec, grid = load_grid(config)
    cadastre = pd.read_parquet(config.path(config.paths.fires_out))
    fires = prepare_fires(cadastre, spec)
    fires = fires[fires["cell_id"] >= 0]

    years = _years(config)
    window = config.history.window_years
    bandwidth = config.history.bandwidth_m
    active = grid["is_trentino"].to_numpy()

    blocks = []
    for year in years:
        past = fires[
            (fires["ignition_date"].dt.year < year)
            & (fires["ignition_date"].dt.year >= year - window)
        ]
        density = kernel_density(
            spec,
            past["x"].to_numpy(dtype="float64"),
            past["y"].to_numpy(dtype="float64"),
            bandwidth,
        )

        mean = density[active].mean()
        blocks.append(
            pd.DataFrame(
                {
                    "cell_id": np.arange(spec.n_cells, dtype="int32"),
                    "year": np.int16(year),
                    DENSITY_COLUMN: (density / mean if mean > 0 else density).astype("float32"),
                }
            )[active]
        )
        logger.info("%d: %d ignition(s) in the trailing %d years", year, len(past), window)

    table = pd.concat(blocks, ignore_index=True)
    validate_history(table, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logger.info("Wrote %s: %d row(s) over %d year(s)", out, len(table), len(years))
    return table


def _years(config: Config) -> list[int]:
    """Every year the feature can be asked about, one past the record so serving has a value."""
    covered = config.date_range.end.year
    if config.meteo.extension_end:
        covered = max(covered, config.meteo.extension_end.year)
    return list(range(config.date_range.start.year, covered + 2))


def validate_history(table: pd.DataFrame, config: Config) -> None:
    negative = int((table[DENSITY_COLUMN] < 0).sum())
    if negative:
        logger.error("%d cell-year(s) carry a negative density", negative)

    missing = int(table[DENSITY_COLUMN].isna().sum())
    if missing:
        logger.error("%d cell-year(s) have no density", missing)

    per_year = table.groupby("year")[DENSITY_COLUMN].mean()
    logger.info(
        "Density mean per year runs %.3f to %.3f over %d year(s)",
        per_year.min(),
        per_year.max(),
        len(per_year),
    )
