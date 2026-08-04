"""Ignition-point extraction on synthetic polygons."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from tfire.fires import extract_ignition_points

CRS = "EPSG:25832"

# A C-shaped polygon: the notch on the right swallows the centroid.
#
#   4 +--------+
#     |        |
#   3 |   +----+
#     |   |
#   1 |   +----+
#     |        |
#   0 +--------+
#     0        4
C_SHAPE = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)])

DISJOINT = MultiPolygon(
    [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(20, 20), (22, 20), (22, 22), (20, 22)]),
    ]
)

DONUT = Polygon(
    [(0, 0), (10, 0), (10, 10), (0, 10)],
    holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]],
)

SQUARE = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


def _series(*polygons: Polygon | MultiPolygon) -> gpd.GeoSeries:
    return gpd.GeoSeries(list(polygons), crs=CRS)


@pytest.mark.parametrize(
    "geometry",
    [C_SHAPE, DISJOINT, DONUT],
    ids=["concave", "disjoint_multipolygon", "ring_with_a_hole"],
)
def test_falls_back_to_representative_point_when_centroid_is_outside(
    geometry: Polygon | MultiPolygon,
) -> None:
    assert not geometry.contains(geometry.centroid)

    points, method = extract_ignition_points(_series(geometry))

    assert method.iloc[0] == "representative_point"
    assert geometry.contains(points.iloc[0])


def test_convex_polygon_uses_the_centroid() -> None:
    points, method = extract_ignition_points(_series(SQUARE))

    assert method.iloc[0] == "centroid"
    assert points.iloc[0].equals(SQUARE.centroid)
    assert SQUARE.contains(points.iloc[0])


def test_mixed_batch_chooses_per_row() -> None:
    points, method = extract_ignition_points(_series(SQUARE, C_SHAPE, SQUARE))

    assert list(method) == ["centroid", "representative_point", "centroid"]
    assert all(
        geom.contains(pt) for geom, pt in zip([SQUARE, C_SHAPE, SQUARE], points, strict=True)
    )


def test_output_preserves_crs_and_index() -> None:
    series = _series(SQUARE, C_SHAPE)
    points, method = extract_ignition_points(series)

    assert points.crs == series.crs
    assert list(points.index) == list(series.index)
    assert list(method.index) == list(series.index)
