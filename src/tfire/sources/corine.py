"""Read the cropped CORINE rasters onto the 500 m grid.

The rasters carry CLC grid codes 1 to 44, not the 3-digit codes: 41 is water bodies,
512 matches nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import rasterio

from tfire.config import Config
from tfire.raster import aligned_window

if TYPE_CHECKING:
    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)


def clc_raster_path(config: Config, year: int) -> Path:
    """Locate a cropped edition, the output of `scripts/crop_corine.py`."""
    if year not in config.corine.editions:
        available = sorted(config.corine.editions)
        raise ValueError(f"No CORINE edition {year} in the config. Available: {available}")

    path = config.path(config.paths.corine_out) / f"clc_{year}.tif"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cropped CORINE edition not found: {path}. Run `python scripts/crop_corine.py` first."
        )
    return path


def read_aligned_blocks(spec: GridSpec, raster_path: Path) -> tuple[npt.NDArray[Any], int]:
    """Read the raster over the grid extent, reshaped to one block of pixels per cell.

    Shape is `(n_rows, factor, n_cols, factor)`, so a reduction over axes 1 and 3 gives
    one value per grid cell.
    """
    with rasterio.open(raster_path) as source:
        if source.nodata is None:
            raise ValueError(f"{raster_path} declares no NODATA value; refusing to guess one")
        nodata = int(source.nodata)

        window, factor = aligned_window(spec, source)
        pixel = source.transform.a
        data = source.read(1, window=window)

    logger.info(
        "Read %s: %dx%d pixels of %.0f m, %d per cell",
        raster_path.name,
        data.shape[1],
        data.shape[0],
        pixel,
        factor * factor,
    )
    return data.reshape(spec.n_rows, factor, spec.n_cols, factor), nodata


def fraction_of_codes(
    blocks: npt.NDArray[Any], nodata: int, codes: list[int]
) -> npt.NDArray[np.float64]:
    """Per-cell share of the given class codes, over the valid pixels only.

    A cell whose pixels are all NODATA gets NaN rather than 0, so partial coverage at
    the edge of the raster cannot pass for a genuine absence of the classes.
    """
    valid = blocks != nodata
    matched = np.isin(blocks, codes) & valid

    n_valid = valid.sum(axis=(1, 3))
    n_matched = matched.sum(axis=(1, 3))

    fraction = np.full(n_valid.shape, np.nan, dtype=np.float64)
    np.divide(n_matched, n_valid, out=fraction, where=n_valid > 0)
    return fraction
