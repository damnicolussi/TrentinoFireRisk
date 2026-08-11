"""Geospatial helpers shared by the raster-reading stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import rasterio
from pyproj import CRS
from rasterio.windows import Window

if TYPE_CHECKING:
    from pathlib import Path

    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)

# how close an offset in pixels has to be to a whole number to count as aligned
_ALIGNMENT_TOLERANCE = 1e-6


def grid_transform(spec: GridSpec) -> tuple[float, float, float, float, float, float]:
    """The grid's own affine transform, in the order Earth Engine and rasterio both take."""
    return (spec.resolution_m, 0.0, spec.xmin, 0.0, -spec.resolution_m, spec.ymax)


def aligned_window(spec: GridSpec, source: rasterio.DatasetReader) -> tuple[Window, int]:
    """The window covering the grid extent, and the pixels per cell along one axis."""
    name = source.name

    transform = source.transform
    if transform.b != 0 or transform.d != 0 or transform.e >= 0:
        raise ValueError(f"{name} is rotated or south-up: {transform}")

    expected_crs = CRS.from_user_input(spec.crs)
    if source.crs is None or CRS.from_user_input(source.crs) != expected_crs:
        raise ValueError(f"{name} is in {source.crs}, expected {spec.crs}")

    pixel = transform.a
    if pixel != -transform.e:
        raise ValueError(f"{name} has non-square pixels: {source.res}")
    if spec.resolution_m % pixel != 0:
        raise ValueError(
            f"{name} pixel size {pixel} does not divide the {spec.resolution_m} m grid resolution"
        )
    factor = int(spec.resolution_m // pixel)

    col_off = (spec.xmin - transform.c) / pixel
    row_off = (transform.f - spec.ymax) / pixel
    for axis, offset in (("column", col_off), ("row", row_off)):
        if abs(offset - round(offset)) > _ALIGNMENT_TOLERANCE:
            raise ValueError(
                f"{name} is offset from the grid by {offset - round(offset):.6f} "
                f"pixel(s) in {axis}; the two grids must share a common lattice"
            )

    window = Window(round(col_off), round(row_off), spec.n_cols * factor, spec.n_rows * factor)
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > source.width
        or window.row_off + window.height > source.height
    ):
        raise ValueError(
            f"{name} does not cover the grid extent: needs {window} "
            f"of a {source.width}x{source.height} raster"
        )

    return window, factor


def read_cell_bands(
    spec: GridSpec, path: Path, expected: tuple[str, ...] | None = None
) -> dict[str, npt.NDArray[np.float64]]:
    """Read a raster already on the grid's own lattice as one flat array per band.

    Arrays come back in `cell_id` order. `expected` asserts the band descriptions, so a stale
    cache written by an earlier version of the producer is caught rather than silently misread.
    """
    with rasterio.open(path) as source:
        _, factor = aligned_window(spec, source)
        if factor != 1:
            raise ValueError(
                f"{path} is at {source.res[0]} m, expected the {spec.resolution_m} m grid"
            )
        names = tuple(source.descriptions)
        if expected is not None and names != expected:
            raise ValueError(
                f"{path} carries bands {names}, expected {expected}. "
                "Delete it and rerun to fetch it again."
            )
        if any(name is None for name in names):
            raise ValueError(f"{path} has unnamed bands; it cannot be read by name")
        data = source.read().astype(np.float64)

    logger.info("Read %d band(s) of %d cell(s) from %s", data.shape[0], spec.n_cells, path.name)
    return dict(zip(names, data.reshape(len(names), spec.n_cells), strict=True))
