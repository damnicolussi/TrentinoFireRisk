"""Geospatial helpers shared by the raster-reading stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import rasterio
from pyproj import CRS
from rasterio.windows import Window

if TYPE_CHECKING:
    from tfire.grid import GridSpec

# how close an offset in pixels has to be to a whole number to count as aligned
_ALIGNMENT_TOLERANCE = 1e-6


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
