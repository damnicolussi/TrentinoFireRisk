"""Credential and connectivity checks for the external data sources."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tfire.config import Config

logger = logging.getLogger(__name__)

# fixed midsummer date so the plausibility bounds below actually mean something
_PROBE_DATE = ("2020", "07", "15")
_PROBE_TIME = "12:00"
_ERA5_DATASET = "reanalysis-era5-land"
_ERA5_VARIABLE = "2m_temperature"
_KELVIN_OFFSET = 273.15
_PLAUSIBLE_TEMP_C = (10.0, 35.0)

_MODIS_COLLECTION = "MODIS/061/MOD13Q1"
_MODIS_NDVI_SCALE = 0.0001  # MODIS NDVI is stored as int16 x 10000
_MODIS_SCALE_M = 250
_MODIS_WINDOW = ("2020-07-01", "2020-07-31")
_PLAUSIBLE_NDVI = (0.3, 0.95)


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


def check_cds(config: Config) -> AccessCheck:
    """Download one hour of ERA5-Land 2m temperature cropped to the province.

    Verifies the API token, that the ERA5-Land license has been accepted,
    server-side cropping, and that the NetCDF backend can open the result.
    """
    import cdsapi
    import xarray as xr

    year, month, day = _PROBE_DATE
    request = {
        "variable": [_ERA5_VARIABLE],
        "year": year,
        "month": month,
        "day": day,
        "time": [_PROBE_TIME],
        "area": config.sources.bbox_wgs84.as_cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    logger.info("Requesting %s from CDS; a queue wait of a few minutes is normal", _ERA5_DATASET)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_probe.nc"
            cdsapi.Client().retrieve(_ERA5_DATASET, request, str(target))
            size_kb = target.stat().st_size / 1024

            with xr.open_dataset(target) as dataset:
                name = next(iter(dataset.data_vars))
                grid = dataset[name].squeeze()
                celsius = float(grid.mean()) - _KELVIN_OFFSET
                shape = tuple(grid.shape)
                cells = int(grid.size)
    except Exception as error:  # noqa: BLE001  any failure here is a failed check
        return AccessCheck("CDS", ok=False, detail=f"{type(error).__name__}: {error}")

    if cells <= 1:
        return AccessCheck(
            "CDS", ok=False, detail=f"expected a grid over the province, got {shape}"
        )

    low, high = _PLAUSIBLE_TEMP_C
    if not low < celsius < high:
        return AccessCheck(
            "CDS", ok=False, detail=f"implausible July midday temperature: {celsius:.1f} C"
        )

    return AccessCheck(
        "CDS",
        ok=True,
        detail=(
            f"{_ERA5_DATASET} {_ERA5_VARIABLE}, {size_kb:.1f} KB, "
            f"grid {shape}, mean {celsius:.1f} C"
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


CHECKS: dict[str, Callable[[Config], AccessCheck]] = {
    "cds": check_cds,
    "gee": check_gee,
    "landsat": check_landsat,
}


def check_access(config: Config, sources: Iterable[str]) -> list[AccessCheck]:
    """Run the named probes and return their results, logging each one."""
    results = [CHECKS[name](config) for name in sources]
    for result in results:
        result.log()
    return results
