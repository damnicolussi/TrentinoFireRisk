"""Credential and connectivity checks for the external data sources."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from tfire.config import Config

logger = logging.getLogger(__name__)

# fixed midsummer date so the plausibility bounds below actually mean something
_PROBE_DATE = ("2020", "07", "15")
_PROBE_TIME = "12:00"
_ERA5_VARIABLE = "2m_temperature"
_KELVIN_OFFSET = 273.15
_PLAUSIBLE_TEMP_C = (10.0, 35.0)

_MODIS_COLLECTION = "MODIS/061/MOD13Q1"
_MODIS_NDVI_SCALE = 0.0001  # MODIS NDVI is stored as int16 x 10000
_MODIS_SCALE_M = 250
_MODIS_WINDOW = ("2020-07-01", "2020-07-31")
_PLAUSIBLE_NDVI = (0.3, 0.95)

_OSM_PROBE_BBOX = (11.115, 46.065, 11.130, 46.080)  # ~1 km around Trento railway station

# a backbone lattice point over the province, and what a day there can plausibly average
_FORECAST_PROBE = (11.1, 46.1)
_PLAUSIBLE_PROBE_TEMP_C = (-30.0, 40.0)
_WORLDPOP_SCALE_M = 100
_PLAUSIBLE_POP_DENSITY = (10.0, 1000.0)  # province average; mountainous and sparse


@dataclass(frozen=True)
class AccessCheck:
    """Outcome of one source probe."""

    source: str
    ok: bool
    detail: str

    def log(self) -> None:
        if self.ok:
            logger.info("%s access OK: %s", self.source, self.detail)
        else:
            logger.error("%s access FAILED: %s", self.source, self.detail)


def check_era5(config: Config) -> AccessCheck:
    """Reduce one midday hour of ERA5-Land 2m temperature over the province.

    Verifies the collection is reachable on the 0.1 degree lattice the fetcher assumes,
    and that the values arrive in kelvin rather than already converted.
    """
    import ee

    from tfire.sources.era5land import (
        GEE_BANDS,
        GEE_COLLECTION,
        RESOLUTION_DEG,
        bbox_lattice,
    )

    lattice = bbox_lattice(config)
    band = GEE_BANDS[_ERA5_VARIABLE]
    stamp = "-".join(_PROBE_DATE)
    try:
        ee.Initialize(project=config.sources.gee_project)
        region = ee.Geometry.Rectangle(config.sources.bbox_wgs84.as_bounds())
        image = (
            ee.ImageCollection(GEE_COLLECTION)
            .filterDate(f"{stamp}T{_PROBE_TIME}", f"{stamp}T{_PROBE_TIME[:2]}:59")
            .select([band])
            .first()
        )
        reduced = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            crs="EPSG:4326",
            scale=RESOLUTION_DEG * 111320,
            maxPixels=int(1e6),
        ).getInfo()
        transform = image.select(band).projection().getInfo()["transform"]
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("ERA5", ok=False, detail=f"{type(error).__name__}: {error}")

    kelvin = reduced.get(band)
    if kelvin is None:
        return AccessCheck("ERA5", ok=False, detail="reduction returned no temperature")

    if abs(transform[0]) != RESOLUTION_DEG or abs(transform[4]) != RESOLUTION_DEG:
        return AccessCheck(
            "ERA5", ok=False, detail=f"expected a {RESOLUTION_DEG} degree grid, got {transform}"
        )

    celsius = float(kelvin) - _KELVIN_OFFSET
    low, high = _PLAUSIBLE_TEMP_C
    if not low < celsius < high:
        return AccessCheck(
            "ERA5", ok=False, detail=f"implausible July midday temperature: {celsius:.1f} C"
        )

    return AccessCheck(
        "ERA5",
        ok=True,
        detail=(
            f"{GEE_COLLECTION} {band}, {lattice.n_rows}x{lattice.n_cols} lattice, "
            f"mean {celsius:.1f} C"
        ),
    )


def check_gee(config: Config) -> AccessCheck:
    """Reduce one MODIS NDVI composite over the province, server-side.

    Nothing is downloaded. Verifies authentication, the Cloud project binding,
    collection access and geometry handling.
    """
    import ee

    project = config.sources.gee_project
    try:
        ee.Initialize(project=project)
        region = ee.Geometry.Rectangle(config.sources.bbox_wgs84.as_bounds())
        start, end = _MODIS_WINDOW

        image = (
            ee.ImageCollection(_MODIS_COLLECTION)
            .filterDate(start, end)
            .filterBounds(region)
            .first()
        )
        image_id = image.getInfo()["id"]
        ndvi = (
            image.select("NDVI")
            .multiply(_MODIS_NDVI_SCALE)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=_MODIS_SCALE_M,
                maxPixels=int(1e9),
            )
            .get("NDVI")
            .getInfo()
        )
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("GEE", ok=False, detail=f"{type(error).__name__}: {error}")

    if ndvi is None:
        return AccessCheck("GEE", ok=False, detail="reduction returned no NDVI value")

    low, high = _PLAUSIBLE_NDVI
    if not low < float(ndvi) < high:
        return AccessCheck(
            "GEE", ok=False, detail=f"implausible peak-summer alpine NDVI: {float(ndvi):.3f}"
        )

    return AccessCheck(
        "GEE",
        ok=True,
        detail=(
            f"project {project!r}, {image_id} at {_MODIS_SCALE_M} m, mean NDVI {float(ndvi):.3f}"
        ),
    )


def check_landsat(config: Config) -> AccessCheck:
    """Count the scenes each configured Landsat collection holds over the province."""
    import ee

    try:
        ee.Initialize(project=config.sources.gee_project)
        region = ee.Geometry.Rectangle(config.sources.bbox_wgs84.as_bounds())
        counts = {
            collection.split("/")[1]: int(
                ee.ImageCollection(collection).filterBounds(region).size().getInfo()
            )
            for collection in config.vegetation.collections
        }
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("Landsat", ok=False, detail=f"{type(error).__name__}: {error}")

    empty = sorted(name for name, count in counts.items() if not count)
    if empty:
        return AccessCheck("Landsat", ok=False, detail=f"no scenes over Trentino for {empty}")

    scenes = ", ".join(f"{name} {count}" for name, count in counts.items())
    return AccessCheck("Landsat", ok=True, detail=f"{sum(counts.values())} scene(s): {scenes}")


def check_osm(config: Config) -> AccessCheck:
    """One small Overpass query near a known landmark (Trento railway station)."""
    import osmnx as ox

    from tfire.sources.osm import _configure_cache

    try:
        _configure_cache(config)
        features = ox.features.features_from_bbox(bbox=_OSM_PROBE_BBOX, tags={"railway": True})
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("OSM", ok=False, detail=f"{type(error).__name__}: {error}")

    if features.empty:
        return AccessCheck("OSM", ok=False, detail="no railway feature near Trento station")
    return AccessCheck("OSM", ok=True, detail=f"{len(features)} feature(s) near Trento station")


def check_worldpop(config: Config) -> AccessCheck:
    """Reduce one WorldPop year over the province bbox, server side."""
    import ee

    try:
        ee.Initialize(project=config.sources.gee_project)
        region = ee.Geometry.Rectangle(config.sources.bbox_wgs84.as_bounds())
        year = config.human.worldpop_years[-1]
        counts = (
            ee.ImageCollection(config.human.worldpop_collection)
            .filter(ee.Filter.eq("year", year))
            .mosaic()
            .select("population")
        )
        total = (
            counts.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=_WORLDPOP_SCALE_M,
                maxPixels=int(1e9),
            )
            .get("population")
            .getInfo()
        )
        area_km2 = region.area().getInfo() / 1e6
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("WorldPop", ok=False, detail=f"{type(error).__name__}: {error}")

    if total is None:
        return AccessCheck("WorldPop", ok=False, detail="reduction returned no population")

    density = float(total) / area_km2
    low, high = _PLAUSIBLE_POP_DENSITY
    if not low < density < high:
        return AccessCheck("WorldPop", ok=False, detail=f"implausible density: {density:.1f}/km2")
    return AccessCheck("WorldPop", ok=True, detail=f"{year}: {density:.1f}/km2 over the bbox")


def check_forecast(config: Config) -> AccessCheck:
    """Ask both Open-Meteo endpoints for one lattice point, and check they still reach."""
    from datetime import date, timedelta

    from tfire.features.meteo import HOURLY_FIELDS
    from tfire.sources.era5land import Lattice
    from tfire.sources.forecast import PROVIDERS, fetch_hourly, window

    today = date.today()
    probe = Lattice(
        latitudes=np.array([_FORECAST_PROBE[1]]), longitudes=np.array([_FORECAST_PROBE[0]])
    )

    reached: list[str] = []
    for provider in PROVIDERS:
        first, last = window(config, provider, today)
        start = max(first, last - timedelta(days=1))
        try:
            fields = fetch_hourly(config, probe, start, last, provider, today)
        except Exception as error:  # noqa: BLE001  any failure here is a failed check
            return AccessCheck("Open-Meteo", ok=False, detail=f"{provider}: {error}")

        empty = [name for name in HOURLY_FIELDS if np.isnan(getattr(fields, name)).all()]
        if empty:
            return AccessCheck(
                "Open-Meteo",
                ok=False,
                detail=f"{provider} carries no {', '.join(empty)} at the edge of its window",
            )

        temperature = float(np.nanmean(fields.temp_c))
        low, high = _PLAUSIBLE_PROBE_TEMP_C
        if not low < temperature < high:
            return AccessCheck(
                "Open-Meteo", ok=False, detail=f"{provider}: implausible {temperature:.1f} C"
            )
        reached.append(f"{provider} reaches {last}")

    return AccessCheck("Open-Meteo", ok=True, detail=", ".join(reached))


CHECKS: dict[str, Callable[[Config], AccessCheck]] = {
    "era5": check_era5,
    "forecast": check_forecast,
    "gee": check_gee,
    "landsat": check_landsat,
    "osm": check_osm,
    "worldpop": check_worldpop,
}


def check_access(config: Config, sources: Iterable[str]) -> list[AccessCheck]:
    """Run the named probes and return their results, logging each one."""
    results = [CHECKS[name](config) for name in sources]
    for result in results:
        result.log()
    return results
