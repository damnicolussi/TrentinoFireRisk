"""Backbone temperature and humidity moved to each cell's own height by a standard lapse rate.

Humidity and vapor pressure deficit are recomputed hourly at fixed temperature offsets on the
backbone, and each cell interpolates between the two offsets around its own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.features.meteo import RUNG_STATISTICS, UNCLIPPED_AT_RUNGS, rung_column

if TYPE_CHECKING:
    from tfire.sources.era5land import Lattice

logger = logging.getLogger(__name__)

OFFSET_COLUMN: Final = "lapse_offset_k"
SUFFIX: Final = "_lapse"


def shifted_columns(config: Config) -> tuple[str, ...]:
    """Temperatures, which move by the offset itself. A range does not move."""
    return (
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_noon",
        f"temp_mean_{config.meteo.temp_window_days}d",
    )


def blended_columns(config: Config) -> tuple[str, ...]:
    """Statistics read between the rungs that bracket a cell's offset."""
    return (*RUNG_STATISTICS, f"rh_mean_{config.meteo.rh_window_days}d")


def lapse_columns(config: Config) -> list[str]:
    """Every column `apply_lapse` adds, in the order the registry declares them."""
    names = [*shifted_columns(config), *blended_columns(config), "rh_range"]
    return [f"{name}{SUFFIX}" for name in names]


def rung_columns(config: Config) -> list[str]:
    return [
        rung_column(name, rung)
        for name in blended_columns(config)
        for rung in config.meteo.lapse_rungs_k
    ]


def cell_offsets(
    elevation_m: npt.NDArray[np.float64],
    backbone_height_m: npt.NDArray[np.float64],
    rate_k_per_km: float,
) -> npt.NDArray[np.float64]:
    """Kelvin to add to the backbone temperature: negative where the cell stands higher."""
    return np.asarray(-rate_k_per_km * (elevation_m - backbone_height_m) / 1000.0)


def build_lapse_offsets(
    config: Config, orography: npt.NDArray[np.float64], weights: pd.DataFrame
) -> pd.DataFrame:
    """One offset per active cell, from its mean height and the bilinear backbone height."""
    pairs = weights.assign(height=orography[weights["era5_id"].to_numpy()])
    pairs["height"] *= pairs["weight"]
    backbone = pairs.groupby("cell_id")["height"].sum()

    topography = pd.read_parquet(config.path(config.paths.topography_out))
    elevation = topography.set_index("cell_id")["elevation_mean"].reindex(backbone.index)
    if elevation.isna().any():
        raise ValueError(f"{int(elevation.isna().sum())} weighted cell(s) carry no elevation")

    offsets = cell_offsets(
        elevation.to_numpy("float64"),
        backbone.to_numpy("float64"),
        config.meteo.lapse_rate_k_per_km,
    )
    return pd.DataFrame({"cell_id": backbone.index.astype("int32"), OFFSET_COLUMN: offsets})


def extract_lapse(config: Config, lattice: Lattice, weights: pd.DataFrame) -> pd.DataFrame:
    """Write `lapse.parquet`, refusing offsets the rungs do not bracket."""
    from tfire.sources.orography import backbone_orography

    orography = backbone_orography(config, lattice)
    frame = build_lapse_offsets(config, orography, weights)

    rungs = config.meteo.lapse_rungs_k
    offsets = frame[OFFSET_COLUMN]
    if offsets.min() < rungs[0] or offsets.max() > rungs[-1]:
        raise ValueError(
            f"Offsets run from {offsets.min():.2f} to {offsets.max():.2f} K, outside the rungs "
            f"{rungs[0]} to {rungs[-1]}; widen meteo.lapse_rungs_k"
        )
    logger.info(
        "Lapse offsets over %d cell(s): %.2f to %.2f K, median %.2f",
        len(frame),
        offsets.min(),
        offsets.max(),
        offsets.median(),
    )

    out = config.path(config.paths.lapse_out)
    frame.to_parquet(out, index=False)
    logger.info("Wrote %s", out)
    return frame


def blend(
    values: npt.NDArray[np.float64],
    rungs: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per row, linear between the two rung columns that bracket the row's offset."""
    if offsets.min() < rungs[0] or offsets.max() > rungs[-1]:
        raise ValueError("An offset lies outside the rungs; nothing is extrapolated")
    lower = np.clip(np.searchsorted(rungs, offsets, side="right") - 1, 0, len(rungs) - 2)
    fraction = (offsets - rungs[lower]) / (rungs[lower + 1] - rungs[lower])
    rows = np.arange(len(offsets))
    return np.asarray(
        (1 - fraction) * values[rows, lower] + fraction * values[rows, lower + 1], dtype="float64"
    )


def apply_lapse(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add the `*_lapse` columns and drop the rungs and the offset they were read from.

    Both the training assembly and the served day call this on the bilinear backbone values,
    so the two paths cannot disagree about it.
    """
    offsets = frame[OFFSET_COLUMN].to_numpy("float64")
    rungs = np.asarray(config.meteo.lapse_rungs_k, dtype="float64")
    added: dict[str, npt.NDArray[np.float64]] = {}

    for name in shifted_columns(config):
        added[f"{name}{SUFFIX}"] = frame[name].to_numpy("float64") + offsets
    for name in blended_columns(config):
        stacked = np.column_stack(
            [frame[rung_column(name, int(rung))].to_numpy("float64") for rung in rungs]
        )
        blended = blend(stacked, rungs, offsets)
        if name in UNCLIPPED_AT_RUNGS:
            blended = np.clip(blended, 0.0, 100.0)
        added[f"{name}{SUFFIX}"] = blended
    added[f"rh_range{SUFFIX}"] = added[f"rh_max{SUFFIX}"] - added[f"rh_min{SUFFIX}"]

    out = frame.drop(columns=[*rung_columns(config), OFFSET_COLUMN])
    for name in lapse_columns(config):
        out[name] = added[name].astype("float32")
    return out
