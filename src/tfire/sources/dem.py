"""Copernicus DEM GLO-30 terrain, aggregated to the 500 m grid inside Earth Engine.

Nothing is downloaded at 30 m: the mean, standard deviation and aspect proportions are
reduced server side and only the 500 m result comes back.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from tfire.config import Config
from tfire.raster import aligned_window

if TYPE_CHECKING:
    import ee

    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)

N_ASPECT_CLASSES: Final = 8

CONTINUOUS_BANDS: Final = ("elevation", "slope", "roughness")
ASPECT_BANDS: Final = ("aspect_NODATA", *(f"aspect_{i + 1}" for i in range(N_ASPECT_CLASSES)))

# what the reducer names its outputs, and what the cached raster calls them
_EE_BAND_ORDER: Final = (
    tuple(f"{band}_{statistic}" for band in CONTINUOUS_BANDS for statistic in ("mean", "stdDev"))
    + ASPECT_BANDS
)
BAND_NAMES: Final = (
    tuple(f"{band}_{statistic}" for band in CONTINUOUS_BANDS for statistic in ("mean", "std"))
    + ASPECT_BANDS
)

_DOWNLOAD_TIMEOUT_S: Final = 900
_RETRY_ATTEMPTS: Final = 3


def aspect_sectors(n: int = N_ASPECT_CLASSES) -> list[tuple[float, float]]:
    """Half-open bearing sectors, class 1 centered on north and wrapping through 0.

    A sector whose start exceeds its end is the one that crosses north; every other
    caller has to handle that case rather than assume `start < end`.
    """
    if n < 1:
        raise ValueError(f"Need at least one aspect sector, got {n}")

    width = 360.0 / n
    return [((i * width - width / 2) % 360.0, (i * width + width / 2) % 360.0) for i in range(n)]


def terrain_image(spec: GridSpec, config: Config) -> ee.Image:
    """The 15-band terrain stack, already reduced onto the grid's own lattice."""
    import ee

    collection = ee.ImageCollection(config.sources.dem_collection).select("DEM")

    # a collection of tiles, so it needs mosaicking; and terrain has to be computed in a
    # metric projection, resampled first or the nearest-neighbor steps show up as slope
    dem = (
        collection.mosaic()
        .setDefaultProjection(collection.first().projection())
        .resample("bilinear")
        .reproject(crs=spec.crs, scale=config.sources.dem_scale_m)
    )
    terrain = ee.Terrain.products(dem)
    elevation = terrain.select("DEM").rename("elevation")
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")

    neighborhood = ee.Kernel.square(radius=1, units="pixels")
    roughness = (
        elevation.reduceNeighborhood(ee.Reducer.max(), neighborhood)
        .subtract(elevation.reduceNeighborhood(ee.Reducer.min(), neighborhood))
        .rename("roughness")
    )

    # aspect is undefined on flat ground, where Earth Engine returns 0 and so cannot be
    # told apart from due north
    flat = slope.eq(0)
    masks = [flat.rename(ASPECT_BANDS[0])]
    for index, (start, end) in enumerate(aspect_sectors()):
        within = (
            aspect.gte(start).And(aspect.lt(end))
            if start < end
            else aspect.gte(start).Or(aspect.lt(end))
        )
        masks.append(within.And(flat.Not()).rename(ASPECT_BANDS[index + 1]))

    continuous = elevation.addBands(slope).addBands(roughness)
    statistics = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)

    # the default of 64 input pixels per output pixel is below the ~289 a 500 m cell holds
    aggregated = continuous.reduceResolution(statistics, maxPixels=1024).addBands(
        ee.Image.cat(masks).reduceResolution(ee.Reducer.mean(), maxPixels=1024)
    )
    image: ee.Image = (
        aggregated.select(list(_EE_BAND_ORDER))
        .rename(list(BAND_NAMES))
        .reproject(crs=spec.crs, crsTransform=list(_grid_transform(spec)))
    )
    return image


def _grid_transform(spec: GridSpec) -> tuple[float, float, float, float, float, float]:
    return (spec.resolution_m, 0.0, spec.xmin, 0.0, -spec.resolution_m, spec.ymax)


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
    # `crs_transform`, not `crsTransform`: with the camel-case spelling Earth Engine
    # ignores the transform and tries to serve the extent at one pixel per metre
    url = image.getDownloadURL(
        {
            "region": region,
            "crs": spec.crs,
            "crs_transform": list(_grid_transform(spec)),
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    return response.content


def _write_cache(payload: bytes, spec: GridSpec, path: Path) -> None:
    """Store the download as a named-band float32 GeoTIFF."""
    with rasterio.open(io.BytesIO(payload)) as source:
        if source.count != len(BAND_NAMES):
            raise ValueError(
                f"Expected {len(BAND_NAMES)} bands from Earth Engine, got {source.count}"
            )
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
        count=len(BAND_NAMES),
        dtype="float32",
        crs=spec.crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as destination:
        destination.write(data)
        destination.descriptions = BAND_NAMES


def fetch_terrain(spec: GridSpec, config: Config) -> Path:
    """Download the terrain stack unless it is already cached."""
    path = config.path(config.paths.dem_raster)
    if path.is_file():
        logger.info("DEM already cached, skipping the Earth Engine request: %s", path)
        return path

    import ee

    ee.Initialize(project=config.sources.gee_project)
    logger.info(
        "Requesting %s at %d m over %d x %d cells; the reduction takes a few minutes",
        config.sources.dem_collection,
        config.sources.dem_scale_m,
        spec.n_cols,
        spec.n_rows,
    )

    payload = _download(terrain_image(spec, config), spec)
    _write_cache(payload, spec, path)
    logger.info("Wrote %d bands to %s (%.1f MB)", len(BAND_NAMES), path, path.stat().st_size / 1e6)
    return path


def read_cell_bands(spec: GridSpec, path: Path) -> dict[str, npt.NDArray[np.float64]]:
    """Read the cached terrain raster as one flat array per band, in `cell_id` order."""
    with rasterio.open(path) as source:
        window, factor = aligned_window(spec, source)
        if factor != 1:
            raise ValueError(
                f"{path} is at {source.res[0]} m, expected the {spec.resolution_m} m grid"
            )
        if tuple(source.descriptions) != BAND_NAMES:
            raise ValueError(
                f"{path} carries bands {source.descriptions}, expected {BAND_NAMES}. "
                "Delete it and rerun to fetch it again."
            )
        data = source.read(window=window).astype(np.float64)

    logger.info("Read %d band(s) of %d cell(s) from %s", data.shape[0], spec.n_cells, path.name)
    return dict(zip(BAND_NAMES, data.reshape(len(BAND_NAMES), spec.n_cells), strict=True))
