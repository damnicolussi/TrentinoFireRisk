"""Grid addressing, geometry lookups and the CORINE block alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon, box

from tfire.grid import GridSpec, build_grid_spec, mark_active
from tfire.sources.corine import fraction_of_codes, read_aligned_blocks

CRS = "EPSG:25832"

# the real province grid, so the test anchors on the shape the pipeline actually builds
TRENTINO = GridSpec(
    xmin=612000.0, ymax=5157500.0, n_cols=234, n_rows=196, resolution_m=500, crs=CRS
)

# 5 cells of 1000 m, cell_id 0 in the north-west corner
SMALL = GridSpec(xmin=0.0, ymax=5000.0, n_cols=5, n_rows=5, resolution_m=1000, crs=CRS)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((612484.81, 5059545.95, 728617.25, 5157140.40), (612000.0, 5157500.0, 234, 196)),
        ((1000.0, 2000.0, 3000.0, 4000.0), (1000.0, 4000.0, 4, 4)),
        ((10.0, 10.0, 20.0, 20.0), (0.0, 500.0, 1, 1)),
    ],
    ids=["province", "already_aligned", "smaller_than_one_cell"],
)
def test_bounds_snap_outward_onto_the_resolution_lattice(
    bounds: tuple[float, float, float, float], expected: tuple[float, float, int, int]
) -> None:
    spec = build_grid_spec(bounds, resolution_m=500, crs=CRS)

    assert (spec.xmin, spec.ymax, spec.n_cols, spec.n_rows) == expected
    assert spec.xmin % 500 == 0 and spec.ymax % 500 == 0
    assert box(*spec.bounds).contains(box(*bounds))
    # snapping an already-snapped extent must not grow it again
    assert build_grid_spec(spec.bounds, resolution_m=500, crs=CRS) == spec


def test_cell_zero_is_the_north_west_corner() -> None:
    first_x, first_y = SMALL.cell_center(0)
    last_x, last_y = SMALL.cell_center(SMALL.n_cells - 1)

    assert (float(first_x), float(first_y)) == (500.0, 4500.0)
    assert (float(last_x), float(last_y)) == (4500.0, 500.0)


def test_every_cell_center_maps_back_to_its_own_cell() -> None:
    cell_id = np.arange(TRENTINO.n_cells)
    x, y = TRENTINO.cell_center(cell_id)

    assert np.array_equal(TRENTINO.points_to_cells(x, y), cell_id)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.0, 5000.0), 0),
        ((1000.0, 5000.0), 1),
        ((0.0, 4000.0), 5),
        ((5000.0, 4500.0), None),
        ((500.0, 0.0), None),
        ((-1.0, 4500.0), None),
    ],
    ids=["origin", "north_edge", "west_edge", "east_bound", "south_bound", "outside"],
)
def test_a_cell_owns_its_west_and_north_edges(
    point: tuple[float, float], expected: int | None
) -> None:
    assert SMALL.point_to_cell(*point) == expected


def test_polygon_to_cells_returns_every_cell_the_geometry_covers() -> None:
    covered = SMALL.polygon_to_cells(box(500, 3500, 2500, 4500))

    assert covered == [0, 1, 2, 5, 6, 7]


def test_polygon_to_cells_keeps_a_cell_clipped_only_at_a_corner() -> None:
    sliver = Polygon([(1, 4999), (2, 4999), (1, 4998)])

    assert SMALL.polygon_to_cells(sliver) == [0]


def test_polygon_to_cells_ignores_the_cells_it_only_touches() -> None:
    # this polygon fills cell 1 exactly and shares an edge or a corner with the five
    # cells around it. Only cell 1 burns.
    assert SMALL.polygon_to_cells(box(1000, 4000, 2000, 5000)) == [1]


def test_polygon_outside_the_grid_covers_nothing() -> None:
    assert SMALL.polygon_to_cells(box(90000, 90000, 91000, 91000)) == []


def test_active_cells_follow_the_boundary_not_the_row_order() -> None:
    boundary = box(0, 0, 2000, 5000)  # the two westernmost columns
    cell_id = np.arange(SMALL.n_cells)
    x, y = SMALL.cell_center(cell_id)

    active = mark_active(boundary, x, y)

    assert set(cell_id[active]) == {row * 5 + col for row in range(5) for col in (0, 1)}


def _write_raster(path: Path, values: np.ndarray, origin: tuple[float, float], pixel: int) -> Path:
    height, width = values.shape
    transform = rasterio.Affine(pixel, 0.0, origin[0], 0.0, -pixel, origin[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int16",
        crs=CRS,
        transform=transform,
        nodata=-128,
    ) as handle:
        handle.write(values, 1)
    return path


# 4x4 pixels of 100 m under a 2x2 grid of 200 m cells, one distinct code per quadrant
QUADRANTS = np.array(
    [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ],
    dtype="int16",
)
QUADRANT_GRID = GridSpec(xmin=0.0, ymax=400.0, n_cols=2, n_rows=2, resolution_m=200, crs=CRS)


def test_blocks_keep_the_north_west_to_south_east_cell_order(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "clc.tif", QUADRANTS, origin=(0.0, 400.0), pixel=100)

    blocks, nodata = read_aligned_blocks(QUADRANT_GRID, raster)

    assert blocks.shape == (2, 2, 2, 2)
    assert nodata == -128
    # code 2 sits in the north-east quadrant, which is cell 1
    assert fraction_of_codes(blocks, nodata, [2]).ravel().tolist() == [0.0, 1.0, 0.0, 0.0]


def test_a_raster_offset_from_the_grid_is_rejected(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "shifted.tif", QUADRANTS, origin=(50.0, 400.0), pixel=100)

    with pytest.raises(ValueError, match="offset from the grid"):
        read_aligned_blocks(QUADRANT_GRID, raster)


def test_a_raster_that_stops_short_of_the_grid_is_rejected(tmp_path: Path) -> None:
    clipped = QUADRANTS[:2, :2]
    raster = _write_raster(tmp_path / "small.tif", clipped, origin=(0.0, 400.0), pixel=100)

    with pytest.raises(ValueError, match="does not cover"):
        read_aligned_blocks(QUADRANT_GRID, raster)


def test_class_fractions_ignore_nodata_instead_of_counting_it_as_absence() -> None:
    pixels = np.array(
        [
            [1, 1, 1, -128],
            [5, 5, -128, -128],
            [-128, -128, 9, 9],
            [-128, -128, 9, 9],
        ],
        dtype="int16",
    )

    fraction = fraction_of_codes(pixels.reshape(2, 2, 2, 2), nodata=-128, codes=[1]).ravel()

    assert fraction[0] == 0.5  # two of four pixels
    assert fraction[1] == 1.0  # the only valid pixel matches
    assert np.isnan(fraction[2])  # no coverage at all, not an absence
    assert fraction[3] == 0.0
