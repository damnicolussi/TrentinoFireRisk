"""OpenStreetMap roads, railways, waterways and points of interest, via osmnx/Overpass."""

from __future__ import annotations

import logging
from typing import Final

import geopandas as gpd
import numpy as np
import osmnx as ox
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from tfire.config import Config

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS: Final = 5
_RETRY_WAIT_MULTIPLIER_S: Final = 20
_RETRY_WAIT_MAX_S: Final = 300

# `highway=True` also pulls in every hiking path, forestry track and footway. Restricted
# to classes a vehicle can use; pedestrian-only trails are already reflected indirectly
# through `recreational_poi_density`.
_VEHICLE_HIGHWAY_VALUES: Final = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
]

LINE_TAGS: Final[dict[str, dict[str, bool | str | list[str]]]] = {
    "roads": {"highway": _VEHICLE_HIGHWAY_VALUES},
    "railways": {"railway": ["rail", "light_rail", "narrow_gauge", "funicular", "tram"]},
    "waterways": {"waterway": ["river", "stream", "canal", "drain", "ditch"]},
}

# one query covers both recreational POIs and accommodation; the spec's own lists overlap
# (a campsite or alpine hut counts toward both), so there is no reason to fetch it twice.
POI_TAGS: Final[dict[str, bool | str | list[str]]] = {"tourism": True, "amenity": "parking"}

RECREATIONAL_TOURISM_VALUES: Final = frozenset(
    {"alpine_hut", "wilderness_hut", "camp_site", "picnic_site", "information"}
)
ACCOMMODATION_TOURISM_VALUES: Final = frozenset(
    {
        "hotel",
        "guest_house",
        "bed_and_breakfast",
        "hostel",
        "motel",
        "camp_site",
        "alpine_hut",
        "wilderness_hut",
        "chalet",
        "apartment",
    }
)


def _configure_cache(config: Config) -> None:
    cache_dir = config.path(config.paths.osm_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.use_cache = True
    ox.settings.requests_timeout = config.human.osm_overpass_timeout_s


def _bbox(config: Config) -> tuple[float, float, float, float]:
    west, south, east, north = config.sources.bbox_wgs84.as_bounds()
    return (west, south, east, north)


@retry(
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=_RETRY_WAIT_MULTIPLIER_S, max=_RETRY_WAIT_MAX_S),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _features_from_bbox(
    bbox: tuple[float, float, float, float], tags: dict[str, bool | str | list[str]]
) -> gpd.GeoDataFrame:
    """Retried on any failure: a dropped connection to Overpass shouldn't lose the whole run.

    osmnx caches each completed sub-request to disk, so a retry mostly resumes rather than
    re-fetching everything from scratch; only the sub-request that actually failed gets
    re-sent.
    """
    return ox.features.features_from_bbox(bbox=bbox, tags=tags)


def fetch_lines(config: Config, kind: str) -> gpd.GeoDataFrame:
    """LineString/MultiLineString features for one OSM line category, in the working CRS."""
    if kind not in LINE_TAGS:
        raise ValueError(f"Unknown OSM line category {kind!r}. Choose from {sorted(LINE_TAGS)}.")

    _configure_cache(config)
    raw = _features_from_bbox(_bbox(config), LINE_TAGS[kind])
    lines = raw.loc[raw.geometry.geom_type.isin(("LineString", "MultiLineString"))]
    logger.info("OSM %s: %d line feature(s) over the bbox", kind, len(lines))
    return gpd.GeoDataFrame(lines[["geometry"]], geometry="geometry", crs=raw.crs).to_crs(
        config.crs
    )


def fetch_poi(config: Config) -> gpd.GeoDataFrame:
    """Tourism and parking points, one point per feature (polygons collapse to a centroid)."""
    _configure_cache(config)
    raw = _features_from_bbox(_bbox(config), POI_TAGS).to_crs(config.crs)

    # centroid computed in the working (projected, metric) CRS, not in raw lat/lon degrees.
    points = gpd.GeoDataFrame(geometry=raw.geometry.centroid, crs=config.crs)
    for tag in ("tourism", "amenity"):
        points[tag] = raw[tag] if tag in raw.columns else np.nan

    logger.info("OSM POI: %d feature(s) over the bbox", len(points))
    return points
