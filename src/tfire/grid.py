"""The 500 m analysis grid: cell identity, geometry lookups and the two masks."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from tfire.config import Config
from tfire.geo import verify_crs
from tfire.sources.corine import clc_raster_path, fraction_of_codes, read_aligned_blocks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridSpec:
    """A north-up rectangular grid, addressed row-major from the top-left corner.

    `y_index` grows southward, matching raster convention, so a cell's block of CORINE
    pixels is a plain reshape rather than a flip. Each cell owns its west and north
    edges; the grid's own east and south edges therefore fall outside it.
    """

    xmin: float
    ymax: float
    n_cols: int
    n_rows: int
    resolution_m: int
    crs: str

    @property
    def n_cells(self) -> int:
        return self.n_cols * self.n_rows

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax."""
        return (
            self.xmin,
            self.ymax - self.n_rows * self.resolution_m,
            self.xmin + self.n_cols * self.resolution_m,
            self.ymax,
        )

    def cell_index(self, cell_id: npt.ArrayLike) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Split a cell id into its column and row index."""
        ids = np.asarray(cell_id)
        return ids % self.n_cols, ids // self.n_cols

    def cell_center(self, cell_id: npt.ArrayLike) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        x_index, y_index = self.cell_index(cell_id)
        return (
            self.xmin + (x_index + 0.5) * self.resolution_m,
            self.ymax - (y_index + 0.5) * self.resolution_m,
        )

    def points_to_cells(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Cell id per point, `-1` where the point falls outside the grid extent."""
        x_index = np.floor((np.asarray(x, dtype=float) - self.xmin) / self.resolution_m)
        y_index = np.floor((self.ymax - np.asarray(y, dtype=float)) / self.resolution_m)

        inside = (x_index >= 0) & (x_index < self.n_cols) & (y_index >= 0) & (y_index < self.n_rows)
        cell_id = y_index.astype(np.int64) * self.n_cols + x_index.astype(np.int64)
        return np.where(inside, cell_id, -1)

    def point_to_cell(self, x: float, y: float) -> int | None:
        cell_id = int(self.points_to_cells([x], [y])[0])
        return None if cell_id < 0 else cell_id

    def polygon_to_cells(self, geom: BaseGeometry) -> list[int]:
        """Every cell whose interior the geometry overlaps.

        A polygon ending exactly on a cell edge does not drag in the cell on the other
        side, which keeps this consistent with `point_to_cell`: both treat a cell as
        half-open.
        """
        minx, miny, maxx, maxy = geom.bounds
        first_col = max(math.floor((minx - self.xmin) / self.resolution_m), 0)
        last_col = min(math.floor((maxx - self.xmin) / self.resolution_m), self.n_cols - 1)
        first_row = max(math.floor((self.ymax - maxy) / self.resolution_m), 0)
        last_row = min(math.floor((self.ymax - miny) / self.resolution_m), self.n_rows - 1)
        if first_col > last_col or first_row > last_row:
            return []

        x_index, y_index = np.meshgrid(
            np.arange(first_col, last_col + 1), np.arange(first_row, last_row + 1)
        )
        candidates = self._boxes(x_index.ravel(), y_index.ravel())

        hit = shapely.intersects(geom, candidates) & ~shapely.touches(geom, candidates)
        cell_id = y_index.ravel() * self.n_cols + x_index.ravel()
        return sorted(int(value) for value in cell_id[hit])

    def cell_polygons(self, cell_id: npt.ArrayLike) -> gpd.GeoSeries:
        x_index, y_index = self.cell_index(cell_id)
        return gpd.GeoSeries(self._boxes(x_index, y_index), crs=self.crs)

    def sub_points(
        self, cell_id: npt.ArrayLike, n_per_side: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """`n_per_side**2` evenly spaced points per cell, shape `(len(cell_id), n_per_side**2)`.

        Offsets stay strictly inside `(0, resolution_m)`, so a sub-point never lands exactly on
        a shared edge between two cells.
        """
        if n_per_side < 1:
            raise ValueError(f"n_per_side must be positive, got {n_per_side}")

        x_index, y_index = self.cell_index(cell_id)
        left = self.xmin + x_index * self.resolution_m
        top = self.ymax - y_index * self.resolution_m

        offset = (np.arange(n_per_side) + 0.5) / n_per_side * self.resolution_m
        dx, dy = np.meshgrid(offset, offset)

        x = left[:, None] + dx.ravel()[None, :]
        y = top[:, None] - dy.ravel()[None, :]
        return x, y

    def _boxes(self, x_index: npt.NDArray[Any], y_index: npt.NDArray[Any]) -> npt.NDArray[Any]:
        left = self.xmin + x_index * self.resolution_m
        top = self.ymax - y_index * self.resolution_m
        boxes: npt.NDArray[Any] = shapely.box(
            left, top - self.resolution_m, left + self.resolution_m, top
        )
        return boxes

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> GridSpec:
        fields: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**fields)


def build_grid_spec(
    bounds: tuple[float, float, float, float], resolution_m: int, crs: str
) -> GridSpec:
    """Snap a bounding box outward onto the resolution lattice.

    Anchoring the origin to a multiple of the resolution is what makes the grid nest
    exactly inside the cropped CORINE rasters, which `scripts/crop_corine.py` snaps the
    same way.
    """
    minx, miny, maxx, maxy = bounds
    left = math.floor(minx / resolution_m) * resolution_m
    bottom = math.floor(miny / resolution_m) * resolution_m
    right = math.ceil(maxx / resolution_m) * resolution_m
    top = math.ceil(maxy / resolution_m) * resolution_m

    return GridSpec(
        xmin=float(left),
        ymax=float(top),
        n_cols=int(round((right - left) / resolution_m)),
        n_rows=int(round((top - bottom) / resolution_m)),
        resolution_m=resolution_m,
        crs=crs,
    )


def load_boundary(config: Config) -> BaseGeometry:
    """The PAT provincial boundary as one geometry, in the working CRS."""
    path = config.path(config.paths.pat_boundary)
    if not path.is_file():
        raise FileNotFoundError(f"PAT boundary not found: {path}")

    boundary = gpd.read_file(path)
    verify_crs(boundary.crs, config.crs, path)

    geometry: BaseGeometry = boundary.geometry.union_all()
    logger.info(
        "Boundary from %s: %d feature(s), %.0f km2", path.name, len(boundary), geometry.area / 1e6
    )
    return geometry


def mark_active(
    boundary: BaseGeometry, x: npt.ArrayLike, y: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """`is_trentino`: the cell center falls inside the boundary.

    Center containment rather than any intersection, so an active cell always has full
    coverage from every raster source. It also lands within one cell of the province
    area divided by the cell area, which the intersects variant does not.
    """
    shapely.prepare(boundary)
    inside: npt.NDArray[np.bool_] = shapely.contains_xy(
        boundary, np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    return inside


def non_burnable_fraction(spec: GridSpec, config: Config) -> npt.NDArray[np.float64]:
    """Per-cell share of CORINE classes that cannot carry a fire."""
    raster = clc_raster_path(config, config.grid.non_burnable_clc_edition)
    blocks, nodata = read_aligned_blocks(spec, raster)
    return fraction_of_codes(blocks, nodata, config.grid.non_burnable_clc_codes).ravel()


def build_grid_frame(spec: GridSpec, boundary: BaseGeometry, config: Config) -> pd.DataFrame:
    cell_id = np.arange(spec.n_cells, dtype=np.int64)
    x_index, y_index = spec.cell_index(cell_id)
    x_coordinate, y_coordinate = spec.cell_center(cell_id)

    is_trentino = mark_active(boundary, x_coordinate, y_coordinate)

    fraction = non_burnable_fraction(spec, config)
    is_non_burnable = np.nan_to_num(fraction, nan=0.0) >= config.grid.non_burnable_threshold

    return pd.DataFrame(
        {
            "cell_id": cell_id.astype("int32"),
            "x_index": x_index.astype("int16"),
            "y_index": y_index.astype("int16"),
            "x_coordinate": x_coordinate.astype("float64"),
            "y_coordinate": y_coordinate.astype("float64"),
            "is_trentino": is_trentino.astype(bool),
            "non_burnable_fraction": fraction.astype("float32"),
            "is_non_burnable": is_non_burnable.astype(bool),
        }
    )


def validate_grid(grid: pd.DataFrame, spec: GridSpec, config: Config) -> None:
    """Check the counts the rest of the pipeline was sized against, logging loudly."""
    active = grid["is_trentino"]
    n_active = int(active.sum())
    expected = config.grid.expected_active_cells

    logger.info(
        "Grid: %d x %d = %d cells at %d m, origin (%.0f, %.0f)",
        spec.n_cols,
        spec.n_rows,
        spec.n_cells,
        spec.resolution_m,
        spec.xmin,
        spec.ymax,
    )
    if n_active != expected:
        logger.error("Active cells: %d, expected %d", n_active, expected)
    else:
        logger.info("Active cells (is_trentino): %d / %d", n_active, spec.n_cells)

    uncovered = int((active & grid["non_burnable_fraction"].isna()).sum())
    if uncovered:
        logger.error("%d active cell(s) have no CORINE coverage", uncovered)

    n_blocked = int((active & grid["is_non_burnable"]).sum())
    logger.info(
        "Non-burnable: %d / %d active (%.2f%%), classes %s at threshold %.2f",
        n_blocked,
        n_active,
        100 * n_blocked / n_active,
        config.grid.non_burnable_clc_codes,
        config.grid.non_burnable_threshold,
    )


def build_grid(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `grid.parquet` and the companion `grid_spec.json`."""
    grid_out = config.path(config.paths.grid_out)
    spec_out = config.path(config.paths.grid_spec_out)

    if grid_out.exists() and spec_out.exists() and not force:
        logger.info("Outputs already exist, skipping (use --force to rebuild): %s", grid_out)
        return pd.read_parquet(grid_out)

    boundary = load_boundary(config)
    spec = build_grid_spec(boundary.bounds, config.resolution_m, config.crs)

    grid = build_grid_frame(spec, boundary, config)
    validate_grid(grid, spec, config)

    grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(grid_out, index=False)
    spec.write(spec_out)
    logger.info("Wrote %d rows to %s", len(grid), grid_out)
    logger.info("Wrote the grid definition to %s", spec_out)

    return grid


def load_grid(config: Config) -> tuple[GridSpec, pd.DataFrame]:
    """Read back what `build_grid` wrote."""
    grid_out = config.path(config.paths.grid_out)
    spec_out = config.path(config.paths.grid_spec_out)
    for path in (grid_out, spec_out):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run `python -m tfire.cli build-grid`.")

    return GridSpec.read(spec_out), pd.read_parquet(grid_out)
