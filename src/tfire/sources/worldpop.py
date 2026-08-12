"""WorldPop population density, aggregated to the 500 m grid inside Earth Engine."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import rasterio
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from tfire.config import Config
from tfire.raster import grid_transform

if TYPE_CHECKING:
    import ee

    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_S: Final = 900
_RETRY_ATTEMPTS: Final = 3


def band_names(config: Config) -> tuple[str, ...]:
    return tuple(f"pop_density_{year}" for year in config.human.worldpop_years)


def population_image(spec: GridSpec, config: Config) -> ee.Image:
    """One band per configured year: mean population density over each 500 m cell.

    Each WorldPop year is served as one image per country, so the collection is mosaicked
    before reducing rather than reduced country by country; countries with no coverage over
    the bbox contribute zero, not a masked gap (see `unmask(0)`).
    """
    import ee

    bands = []
    for year in config.human.worldpop_years:
        collection = ee.ImageCollection(config.human.worldpop_collection).filter(
            ee.Filter.eq("year", year)
        )
        counts = (
            collection.mosaic()
            .unmask(0)
            .setDefaultProjection(collection.first().projection())
            .select("population")
        )
        density_native = counts.divide(ee.Image.pixelArea().divide(1e6))
        density = density_native.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
        bands.append(density.rename(f"pop_density_{year}"))

    image: ee.Image = ee.Image.cat(bands).reproject(
        crs=spec.crs, crsTransform=list(grid_transform(spec))
    )
    return image


@retry(
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=10, max=120),
    reraise=True,
)
def _download(image: ee.Image, spec: GridSpec) -> bytes:
    import ee

    xmin, ymin, xmax, ymax = spec.bounds
    region = ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax], proj=ee.Projection(spec.crs), geodesic=False
    )
    url = image.getDownloadURL(
        {
            "region": region,
            "crs": spec.crs,
            "crs_transform": list(grid_transform(spec)),
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    return response.content


def _write_cache(payload: bytes, spec: GridSpec, path: Path, names: tuple[str, ...]) -> None:
    """Store the download as a named-band float32 GeoTIFF, one band per year."""
    with rasterio.open(io.BytesIO(payload)) as source:
        if source.count != len(names):
            raise ValueError(f"Expected {len(names)} band(s) from Earth Engine, got {source.count}")
        if (source.width, source.height) != (spec.n_cols, spec.n_rows):
            raise ValueError(
                f"Earth Engine returned {source.width}x{source.height}, "
                f"expected the grid's {spec.n_cols}x{spec.n_rows}"
            )
        data = source.read().astype("float32")
        transform = source.transform

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=spec.n_rows,
        width=spec.n_cols,
        count=len(names),
        dtype="float32",
        crs=spec.crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as destination:
        destination.write(data)
        destination.descriptions = names


def fetch_population(spec: GridSpec, config: Config) -> Path:
    """Download the population density stack unless it is already cached."""
    path = config.path(config.paths.worldpop_raster)
    if path.is_file():
        logger.info("WorldPop already cached, skipping the Earth Engine request: %s", path)
        return path

    import ee

    ee.Initialize(project=config.sources.gee_project)
    names = band_names(config)
    logger.info(
        "Requesting %s for years %s at %d m over %d x %d cells",
        config.human.worldpop_collection,
        config.human.worldpop_years,
        spec.resolution_m,
        spec.n_cols,
        spec.n_rows,
    )

    payload = _download(population_image(spec, config), spec)
    _write_cache(payload, spec, path, names)
    logger.info("Wrote %d band(s) to %s (%.1f MB)", len(names), path, path.stat().st_size / 1e6)
    return path
