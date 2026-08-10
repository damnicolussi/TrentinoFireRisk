"""Elevation, slope, roughness and aspect proportions per cell."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from tfire.config import Config
from tfire.grid import load_grid
from tfire.sources.dem import ASPECT_BANDS, BAND_NAMES, fetch_terrain, read_cell_bands

logger = logging.getLogger(__name__)

_SUM_TOLERANCE: Final = 1e-6
_MAX_SLOPE_DEGREES: Final = 90.0


def build_topography_frame(config: Config) -> pd.DataFrame:
    spec, grid = load_grid(config)
    bands = read_cell_bands(spec, fetch_terrain(spec, config))

    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()
    frame = pd.DataFrame({name: bands[name] for name in BAND_NAMES}).astype("float32")
    frame.insert(0, "cell_id", np.arange(spec.n_cells, dtype="int32"))
    return frame.loc[frame["cell_id"].isin(active)].reset_index(drop=True)


def validate_topography(topography: pd.DataFrame, config: Config) -> None:
    """Check the acceptance criteria, logging loudly."""
    missing = int(topography.isna().any(axis=1).sum())
    if missing:
        logger.error("%d active cell(s) have no DEM coverage", missing)

    aspect = topography[list(ASPECT_BANDS)].sum(axis=1)
    deviation = float((aspect.dropna() - 1.0).abs().max())
    if deviation > _SUM_TOLERANCE:
        logger.error("Aspect proportions deviate from 1.0 by up to %.3g", deviation)
    else:
        logger.info(
            "Aspect proportions sum to 1.0 within %.3g, %.2f%% of the cell area flat",
            deviation,
            100 * float(topography["aspect_NODATA"].mean()),
        )

    elevation = topography["elevation_mean"]
    low, high = config.topography.min_elevation_m, config.topography.max_elevation_m
    outside = int((~elevation.between(low, high)).sum())
    if outside:
        logger.error(
            "%d cell(s) outside the plausible elevation range %.0f to %.0f m", outside, low, high
        )
    logger.info(
        "Elevation: %.0f to %.0f m, mean %.0f | slope: mean %.1f deg, max %.1f | "
        "roughness: mean %.1f m",
        elevation.min(),
        elevation.max(),
        elevation.mean(),
        topography["slope_mean"].mean(),
        topography["slope_mean"].max(),
        topography["roughness_mean"].mean(),
    )

    slope = topography["slope_mean"]
    if not slope.between(0, _MAX_SLOPE_DEGREES).all():
        logger.error("slope_mean outside [0, %.0f] degrees", _MAX_SLOPE_DEGREES)


def extract_topography(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `topography.parquet`, one row per active cell."""
    out = config.path(config.paths.topography_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    topography = build_topography_frame(config)
    validate_topography(topography, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    topography.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(topography), topography.shape[1], out)
    return topography
