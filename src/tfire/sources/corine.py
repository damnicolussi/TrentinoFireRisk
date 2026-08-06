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
from pyproj import CRS
from rasterio.windows import Window

from tfire.config import Config

if TYPE_CHECKING:
    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)

# how close an offset in pixels has to be to a whole number to count as aligned
_ALIGNMENT_TOLERANCE = 1e-6


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
    one value per grid cell. Every misfit between the two grids raises: an offset of a
    single pixel would shift the result over the whole province while still looking
    like a plausible map.
    """
    with rasterio.open(raster_path) as source:
        if source.nodata is None:
            raise ValueError(f"{raster_path} declares no NODATA value; refusing to guess one")
        nodata = int(source.nodata)

        transform = source.transform
        if transform.b != 0 or transform.d != 0 or transform.e >= 0:
            raise ValueError(f"{raster_path} is rotated or south-up: {transform}")

        expected_crs = CRS.from_user_input(spec.crs)
        if source.crs is None or CRS.from_user_input(source.crs) != expected_crs:
            raise ValueError(f"{raster_path} is in {source.crs}, expected {spec.crs}")

        pixel = transform.a
        if pixel != -transform.e:
            raise ValueError(f"{raster_path} has non-square pixels: {source.res}")
        if spec.resolution_m % pixel != 0:
            raise ValueError(
                f"{raster_path} pixel size {pixel} does not divide the "
                f"{spec.resolution_m} m grid resolution"
            )
        factor = int(spec.resolution_m // pixel)

        col_off = (spec.xmin - transform.c) / pixel
        row_off = (transform.f - spec.ymax) / pixel
        for name, offset in (("column", col_off), ("row", row_off)):
            if abs(offset - round(offset)) > _ALIGNMENT_TOLERANCE:
                raise ValueError(
                    f"{raster_path} is offset from the grid by {offset - round(offset):.6f} "
                    f"pixel(s) in {name}; the two grids must share a common lattice"
                )

        window = Window(round(col_off), round(row_off), spec.n_cols * factor, spec.n_rows * factor)
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > source.width
            or window.row_off + window.height > source.height
        ):
            raise ValueError(
                f"{raster_path} does not cover the grid extent: needs {window} "
                f"of a {source.width}x{source.height} raster"
            )

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
