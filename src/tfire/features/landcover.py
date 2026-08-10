"""CORINE class proportions per cell, one row per cell and edition."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import GridSpec, load_grid
from tfire.sources.corine import clc_raster_path, read_aligned_blocks

logger = logging.getLogger(__name__)

# CLC grid code (the value stored in the raster) to the 3-digit nomenclature code.
CLC_LEVEL_3: Final[dict[int, int]] = {
    1: 111,
    2: 112,
    3: 121,
    4: 122,
    5: 123,
    6: 124,
    7: 131,
    8: 132,
    9: 133,
    10: 141,
    11: 142,
    12: 211,
    13: 212,
    14: 213,
    15: 221,
    16: 222,
    17: 223,
    18: 231,
    19: 241,
    20: 242,
    21: 243,
    22: 244,
    23: 311,
    24: 312,
    25: 313,
    26: 321,
    27: 322,
    28: 323,
    29: 324,
    30: 331,
    31: 332,
    32: 333,
    33: 334,
    34: 335,
    35: 411,
    36: 412,
    37: 421,
    38: 422,
    39: 423,
    40: 511,
    41: 512,
    42: 521,
    43: 522,
    44: 523,
}

N_CLASSES: Final = len(CLC_LEVEL_3)

# the nomenclature's own no-data class, distinct from the raster's NODATA value. It marks
# sea and land outside the CLC coverage, and never appears over Trentino.
CLC_NODATA_CODE: Final = 48

_SUM_TOLERANCE: Final = 1e-6


def group_by_level(level: int) -> dict[int, list[int]]:
    """Grid codes grouped by the leading digits of their 3-digit code."""
    if level not in (1, 2):
        raise ValueError(f"CLC aggregates exist at level 1 and 2, not {level}")

    divisor = 10 ** (3 - level)
    groups: dict[int, list[int]] = {}
    for grid_code, code in sorted(CLC_LEVEL_3.items()):
        groups.setdefault(code // divisor, []).append(grid_code)
    return groups


def class_proportions(blocks: npt.NDArray[Any], nodata: int) -> npt.NDArray[np.float64]:
    """Per-cell share of every CLC grid code, over the valid pixels only."""
    n_rows, factor, n_cols, _ = blocks.shape
    n_cells = n_rows * n_cols
    per_cell = factor * factor

    flat = blocks.transpose(0, 2, 1, 3).reshape(n_cells, per_cell)
    valid = (flat != nodata) & (flat != CLC_NODATA_CODE)

    seen = np.unique(flat[valid])
    unknown = seen[(seen < 1) | (seen > N_CLASSES)]
    if unknown.size:
        raise ValueError(f"Raster holds values outside CLC grid codes 1-{N_CLASSES}: {unknown}")

    # bincount over cell * (N_CLASSES + 1) + code; NODATA is parked in the unused
    # code 0 column, which is dropped after the reshape.
    code = np.where(valid, flat, 0).astype(np.int64).ravel()
    cell = np.repeat(np.arange(n_cells, dtype=np.int64), per_cell)
    counts = np.bincount(
        cell * (N_CLASSES + 1) + code, minlength=n_cells * (N_CLASSES + 1)
    ).reshape(n_cells, N_CLASSES + 1)[:, 1:]

    n_valid = valid.sum(axis=1)
    proportions = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(counts, n_valid[:, None], out=proportions, where=n_valid[:, None] > 0)
    return proportions


def proportion_frame(proportions: npt.NDArray[np.float64]) -> pd.DataFrame:
    """The 44 class columns plus their level-2 and level-1 aggregates."""
    columns = {f"CLC_{code}": proportions[:, code - 1] for code in sorted(CLC_LEVEL_3)}
    for level in (2, 1):
        for parent, members in group_by_level(level).items():
            index = [code - 1 for code in members]
            columns[f"CLC_L{level}_{parent}"] = proportions[:, index].sum(axis=1)

    return pd.DataFrame(columns).astype("float32")


def nearest_edition(dates: pd.Series, editions: Iterable[int]) -> pd.Series:
    """CORINE edition nearest in time to each date, ties going to the earlier edition."""
    years = np.asarray(sorted(editions), dtype=np.int64)
    if not years.size:
        raise ValueError("No CORINE editions to choose from")

    year = pd.DatetimeIndex(dates).year.to_numpy()
    distance = np.abs(year[:, None] - years[None, :])
    return pd.Series(years[distance.argmin(axis=1)], index=dates.index, dtype="int16")


def edition_frame(
    spec: GridSpec, config: Config, year: int, active: npt.NDArray[Any]
) -> pd.DataFrame:
    """Class proportions for one edition, restricted to the active cells."""
    blocks, nodata = read_aligned_blocks(spec, clc_raster_path(config, year))
    frame = proportion_frame(class_proportions(blocks, nodata))

    frame.insert(0, "cell_id", np.arange(spec.n_cells, dtype="int32"))
    frame.insert(1, "clc_edition", pd.Series(year, index=frame.index, dtype="int16"))
    return frame.loc[frame["cell_id"].isin(active)].reset_index(drop=True)


def validate_landcover(landcover: pd.DataFrame) -> None:
    """Check the acceptance criteria, logging loudly."""
    classes = [f"CLC_{code}" for code in sorted(CLC_LEVEL_3)]

    for year, edition in landcover.groupby("clc_edition"):
        total = edition[classes].sum(axis=1)
        uncovered = int(total.isna().sum())
        if uncovered:
            logger.error("CLC%d: %d active cell(s) have no CORINE coverage", year, uncovered)

        deviation = float((total.dropna() - 1.0).abs().max())
        if deviation > _SUM_TOLERANCE:
            logger.error("CLC%d: class proportions deviate from 1.0 by up to %.3g", year, deviation)
        else:
            logger.info(
                "CLC%d: %d cell(s), proportions sum to 1.0 within %.3g, %d class(es) present",
                year,
                len(edition),
                deviation,
                int((edition[classes].to_numpy() > 0).any(axis=0).sum()),
            )

        for level in (2, 1):
            aggregates = [f"CLC_L{level}_{parent}" for parent in group_by_level(level)]
            gap = float((edition[aggregates].sum(axis=1) - total).abs().max())
            if gap > _SUM_TOLERANCE:
                logger.error(
                    "CLC%d: level-%d aggregates miss the class total by up to %.3g",
                    year,
                    level,
                    gap,
                )


def extract_landcover(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `landcover.parquet`, one row per active cell and CORINE edition."""
    out = config.path(config.paths.landcover_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    spec, grid = load_grid(config)
    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()

    landcover = pd.concat(
        [edition_frame(spec, config, year, active) for year in sorted(config.corine.editions)],
        ignore_index=True,
    )
    validate_landcover(landcover)

    out.parent.mkdir(parents=True, exist_ok=True)
    landcover.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(landcover), landcover.shape[1], out)
    return landcover
