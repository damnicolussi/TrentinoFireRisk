"""Main roads from the OSM cache, and observed against expected ignitions along them."""

from __future__ import annotations

from typing import Any, Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry

from tfire.config import Config

ROADS: Final = ("SS349", "SS612", "SS42")
MAJOR_CLASSES: Final = ("motorway", "trunk", "primary", "secondary")

# a road's corridor is the cells whose center lies within this distance of it
CORRIDOR_M: Final = 1000.0


def major_roads(config: Config) -> gpd.GeoDataFrame:
    from tfire.sources.osm import LINE_TAGS, _bbox, _configure_cache, _features_from_bbox

    _configure_cache(config)
    raw = _features_from_bbox(_bbox(config), LINE_TAGS["roads"])
    kept = raw[raw["highway"].str.removesuffix("_link").isin(MAJOR_CLASSES)]
    lines = kept[kept.geometry.geom_type.isin(("LineString", "MultiLineString"))]
    return gpd.GeoDataFrame(lines[["highway", "ref"]], geometry=lines.geometry, crs=raw.crs).to_crs(
        config.crs
    )


def road_geometry(roads: gpd.GeoDataFrame, ref: str) -> BaseGeometry:
    # a way can carry two refs where routes share a carriageway, "SS349;SS350"
    tokens = roads["ref"].fillna("").str.split(";")
    return roads[tokens.apply(lambda refs: ref in refs)].geometry.union_all()


def poisson_p(observed: int, expected: float) -> float:
    """Two-sided exact Poisson p-value, as twice the smaller tail."""
    from scipy.stats import poisson

    low = poisson.cdf(observed, expected)
    high = poisson.sf(observed - 1, expected)
    return float(min(1.0, 2 * min(low, high)))


def observed_against_expected(
    distance: npt.NDArray[np.float64],
    positions: npt.NDArray[np.int64],
    years: npt.NDArray[np.int64],
    mass: npt.NDArray[np.float64],
    first_test_year: int,
) -> dict[str, Any]:
    recent = years >= first_test_year
    inside = distance[positions] < CORRIDOR_M
    observed = int((inside & recent).sum())
    expected = float(mass[distance < CORRIDOR_M].sum() / mass.sum() * recent.sum())
    record = float(inside.sum() / len(positions) / ((distance < CORRIDOR_M).mean()))
    return {
        "expected": expected,
        "observed": observed,
        "p": poisson_p(observed, expected),
        "record_density": record,
    }


def corridors(config: Config, xy: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.bool_]]:
    """Per road in ROADS, which of the points at `xy` lie within CORRIDOR_M of it."""
    points = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=config.crs)
    roads = major_roads(config)
    return {
        road: points.distance(road_geometry(roads, road)).to_numpy() < CORRIDOR_M for road in ROADS
    }
