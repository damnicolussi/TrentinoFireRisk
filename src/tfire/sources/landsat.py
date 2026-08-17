"""Landsat Collection 2 surface reflectance, composited by month onto the 500 m grid.

Nothing is downloaded at 30 m: cloud masking, index computation and the monthly median all run
inside Earth Engine, and only the 500 m result comes back.
"""

from __future__ import annotations

import io
import logging
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from tfire.config import Config
from tfire.raster import grid_transform

if TYPE_CHECKING:
    import ee

    from tfire.grid import GridSpec

logger = logging.getLogger(__name__)

# ndwi is the NIR/SWIR (Gao) form, canopy water content, not the green/NIR index that maps
# open water
INDEX_BANDS: Final = {
    "ndvi": ("nir", "red"),
    "ndwi": ("nir", "swir1"),
    "nbr": ("nir", "swir2"),
}
INDEX_NAMES: Final = tuple(INDEX_BANDS)
OUTPUT_BANDS: Final = (*INDEX_NAMES, "lst", "valid_fraction", "snow_fraction")

# reflectance aliases in a fixed order, so the two sensor generations can be renamed onto a
# common vocabulary and everything downstream stops caring which satellite it came from
REFLECTANCE_ALIASES: Final = ("blue", "green", "red", "nir", "swir1", "swir2")
_TM_BANDS: Final = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6")
_OLI_BANDS: Final = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10")

# Collection 2 Level-2 digital numbers to physical units
_REFLECTANCE_SCALE: Final = 2.75e-5
_REFLECTANCE_OFFSET: Final = -0.2
_TEMPERATURE_SCALE: Final = 0.00341802
_TEMPERATURE_OFFSET: Final = 149.0
_KELVIN_TO_C: Final = 273.15

# QA_PIXEL bit positions. A pixel is usable only when none of these are set.
_FILL_BIT: Final = 0
_DILATED_CLOUD_BIT: Final = 1
_CIRRUS_BIT: Final = 2
_CLOUD_BIT: Final = 3
_CLOUD_SHADOW_BIT: Final = 4
_SNOW_BIT: Final = 5
_CLOUD_BITS: Final = (_FILL_BIT, _DILATED_CLOUD_BIT, _CIRRUS_BIT, _CLOUD_BIT, _CLOUD_SHADOW_BIT)

# Roy et al. 2016 (RSE 185:57-70) Table 2, OLS: ETM+ reflectance onto the OLI scale. OLI is the
# reference because that is the direction the published coefficients run, so nothing is inverted.
# TM is transformed with the same numbers, the usual treatment of TM and ETM+ as one response.
# Characterized on Collection 1; the Collection 2 processing change is small against the
# inter-sensor difference these correct for.
_ROY_SLOPES: Final = {
    "blue": 0.8474,
    "green": 0.8483,
    "red": 0.9047,
    "nir": 0.8462,
    "swir1": 0.8937,
    "swir2": 0.9071,
}
_ROY_INTERCEPTS: Final = {
    "blue": 0.0003,
    "green": 0.0088,
    "red": 0.0061,
    "nir": 0.0412,
    "swir1": 0.0254,
    "swir2": 0.0172,
}

_NATIVE_SCALE_M: Final = 30
# WGS84 / UTM 32N, what the scenes covering Trentino are delivered in. Aggregating in it rather
# than in the project CRS avoids reprojecting the full extent at 30 m;
_NATIVE_CRS: Final = "EPSG:32632"

# a 500 m cell spans 16.7 of the 30 m pixels, so at worst 18 along each axis once the two
# lattices are offset. The default of 64 is too low;
_MAX_INPUT_PIXELS: Final = 18 * 18

# how the bands split when one month is still too heavy for a single request: the reflectance
# indices share one pass over the scenes, the thermal band and the two coverage fractions another
_BAND_GROUPS: Final = (INDEX_NAMES, ("lst", "valid_fraction", "snow_fraction"))
_DOWNLOAD_TIMEOUT_S: Final = 1800
_RETRY_ATTEMPTS: Final = 3

# Earth Engine says no to a synchronous request that is too heavy rather than failing outright,
# and the fix is a smaller chunk, not another attempt at the same one.
_OVERSIZED_MARKERS: Final = (
    "computation timed out",
    "user memory limit exceeded",
    "too many concurrent aggregations",
    "request payload size exceeds the limit",
    "total request size",
)


@dataclass(frozen=True)
class SensorSpec:
    """How one Landsat collection's bands map onto the common vocabulary."""

    collection: str
    bands: tuple[str, ...]
    harmonize: bool


def sensor_spec(collection: str) -> SensorSpec:
    """Resolve a collection id to its band layout, by the satellite code in the id."""
    satellite = collection.split("/")[1] if "/" in collection else collection
    if satellite in ("LT04", "LT05", "LE07"):
        return SensorSpec(collection, _TM_BANDS, harmonize=True)
    if satellite in ("LC08", "LC09"):
        return SensorSpec(collection, _OLI_BANDS, harmonize=False)
    raise ValueError(
        f"Unknown Landsat collection {collection!r}: expected one of "
        "LT04, LT05, LE07, LC08, LC09 in the second path segment"
    )


def reject_mask(mask_snow: bool) -> int:
    """The QA_PIXEL bits that disqualify a pixel, combined into one mask."""
    bits = (*_CLOUD_BITS, _SNOW_BIT) if mask_snow else _CLOUD_BITS
    return sum(1 << bit for bit in bits)


def month_starts(start: date, end: date) -> list[date]:
    """Every month in the range, as its first day."""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def chunk_starts(config: Config) -> list[date]:
    """The first month of every fetch chunk, over the whole configured record."""
    size = config.vegetation.chunk_months
    first = date(config.date_range.start.year, 1, 1)
    last = date(config.date_range.end.year, 12, 1)
    return [month for month in month_starts(first, last) if (month.month - 1) % size == 0]


def chunk_cache_path(config: Config, chunk: date) -> Path:
    return config.path(config.paths.landsat_raw) / f"{chunk.year}-{chunk.month:02d}.tif"


def band_name(name: str, month: date) -> str:
    return f"{name}_{month.year}{month.month:02d}"


def parse_band_name(description: str) -> tuple[str, date]:
    """Undo `band_name`. Raises rather than guessing, so a stale cache cannot be misread."""
    name, _, stamp = description.rpartition("_")
    if name not in OUTPUT_BANDS or len(stamp) != 6 or not stamp.isdigit():
        raise ValueError(f"Not a Landsat composite band description: {description!r}")
    return name, date(int(stamp[:4]), int(stamp[4:]), 1)


def chunk_months(config: Config, chunk: date) -> list[date]:
    """The months inside one chunk, clipped to the configured record."""
    size = config.vegetation.chunk_months
    year, month = chunk.year, chunk.month
    months = []
    for offset in range(size):
        step = month + offset
        months.append(date(year + (step - 1) // 12, (step - 1) % 12 + 1, 1))
    first = date(config.date_range.start.year, 1, 1)
    last = date(config.date_range.end.year, 12, 1)
    return [value for value in months if first <= value <= last]


def _month_end(month: date) -> date:
    return date(month.year, month.month, monthrange(month.year, month.month)[1])


def _scene_collection(config: Config, region: ee.Geometry, month: date) -> ee.ImageCollection:
    """Every usable scene in the month, on the common band vocabulary and the OLI scale."""
    import ee

    start = ee.Date(month.isoformat())
    end = ee.Date(_month_end(month).isoformat()).advance(1, "day")
    rejected = reject_mask(config.vegetation.mask_snow)
    observed = reject_mask(mask_snow=False)

    def prepare(spec: SensorSpec) -> ee.ImageCollection:
        source = (
            ee.ImageCollection(spec.collection)
            .filterBounds(region)
            .filterDate(start, end)
            .select([*spec.bands, "QA_PIXEL"], [*REFLECTANCE_ALIASES, "thermal", "QA_PIXEL"])
        )

        def one(image: ee.Image) -> ee.Image:
            qa = image.select("QA_PIXEL").toInt()
            clear = qa.bitwiseAnd(rejected).eq(0).rename("clear")
            # snow is counted among pixels that were actually seen, so a cloudy month does not
            # read as snow-free
            seen = qa.bitwiseAnd(observed).eq(0)
            snow = qa.rightShift(_SNOW_BIT).bitwiseAnd(1).updateMask(seen).rename("snow")

            reflectance = (
                image.select(list(REFLECTANCE_ALIASES))
                .multiply(_REFLECTANCE_SCALE)
                .add(_REFLECTANCE_OFFSET)
            )
            if spec.harmonize:
                slopes = [_ROY_SLOPES[name] for name in REFLECTANCE_ALIASES]
                intercepts = [_ROY_INTERCEPTS[name] for name in REFLECTANCE_ALIASES]
                reflectance = reflectance.multiply(ee.Image.constant(slopes)).add(
                    ee.Image.constant(intercepts)
                )

            lst = (
                image.select("thermal")
                .multiply(_TEMPERATURE_SCALE)
                .add(_TEMPERATURE_OFFSET - _KELVIN_TO_C)
                .rename("lst")
            )
            values = _spectral_indices(reflectance).addBands(lst).updateMask(clear)
            return values.addBands(clear).addBands(snow).toFloat()

        prepared: ee.ImageCollection = source.map(one)
        return prepared

    collections = [prepare(sensor_spec(name)) for name in config.vegetation.collections]
    merged: ee.ImageCollection = collections[0]
    for extra in (*collections[1:], ee.ImageCollection([_placeholder()])):
        merged = merged.merge(extra)
    return merged


def _placeholder() -> ee.Image:
    """A fully masked scene, so an empty month still reduces to the right band names."""
    import ee

    names = [*INDEX_NAMES, "lst", "clear", "snow"]
    blank = ee.Image.constant([0.0] * len(names)).rename(names).toFloat()
    # the placeholder's projection is what an otherwise-empty month reduces in, so it has to be
    # the scenes' own, not the default degrees
    masked: ee.Image = blank.updateMask(ee.Image.constant(0)).setDefaultProjection(
        crs=_NATIVE_CRS, scale=_NATIVE_SCALE_M
    )
    return masked


def _spectral_indices(reflectance: ee.Image) -> ee.Image:
    """The indices, computed per scene so the monthly composite averages indices, not bands.

    `normalizedDifference` rather than `expression`: an expression re-evaluates the whole masked
    reflectance stack per call, and two of them in one request exhaust Earth Engine's memory.
    """
    computed = [
        reflectance.normalizedDifference(list(pair)).rename(name)
        for name, pair in INDEX_BANDS.items()
    ]
    image: ee.Image = computed[0]
    for extra in computed[1:]:
        image = image.addBands(extra)
    return image


def month_image(
    config: Config,
    spec: GridSpec,
    region: ee.Geometry,
    month: date,
    bands: tuple[str, ...] = OUTPUT_BANDS,
) -> ee.Image:
    """One month's composite, reduced onto the grid's own lattice.

    Only the requested bands are computed, so a month too heavy for one request can be fetched
    as two lighter ones.
    """
    import ee

    scenes = _scene_collection(config, region, month)
    parts = []
    measured = [name for name in bands if name in (*INDEX_NAMES, "lst")]
    if measured:
        parts.append(scenes.select(measured).median())
    if "valid_fraction" in bands:
        # a pixel counts as observed if any scene in the month saw it clear; the cell's fraction
        # is then what tells the model how much of it the composite actually describes
        parts.append(scenes.select("clear").max().unmask(0).rename("valid_fraction"))
    if "snow_fraction" in bands:
        parts.append(scenes.select("snow").mean().unmask(0).rename("snow_fraction"))

    combined: ee.Image = parts[0]
    for extra in parts[1:]:
        combined = combined.addBands(extra)

    # aggregate in the scenes' own projection: forcing the full extent through a 30 m
    # reprojection into the project CRS first costs three times as much and buys nothing, since
    # the result lands on the grid lattice below either way
    native = combined.select(list(bands)).setDefaultProjection(
        crs=_NATIVE_CRS, scale=_NATIVE_SCALE_M
    )
    aggregated: ee.Image = native.reduceResolution(
        ee.Reducer.mean(), maxPixels=_MAX_INPUT_PIXELS
    ).reproject(crs=spec.crs, crsTransform=list(grid_transform(spec)))
    return aggregated.rename([band_name(name, month) for name in bands])


def window_image(
    config: Config, spec: GridSpec, months: list[date], bands: tuple[str, ...] = OUTPUT_BANDS
) -> ee.Image:
    """One request's worth of months, as a single stack of named bands."""
    import ee

    xmin, ymin, xmax, ymax = spec.bounds
    region = ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax], proj=ee.Projection(spec.crs), geodesic=False
    )
    image: ee.Image = month_image(config, spec, region, months[0], bands)
    for month in months[1:]:
        image = image.addBands(month_image(config, spec, region, month, bands))
    return image


def is_oversized(error: BaseException) -> bool:
    """Whether Earth Engine refused the request for its weight rather than its content."""
    response = getattr(error, "response", None)
    body = getattr(response, "text", "") if response is not None else ""
    reason = f"{error} {body}".lower()
    return any(marker in reason for marker in _OVERSIZED_MARKERS)


@retry(
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=10, max=120),
    retry=retry_if_exception(lambda error: not is_oversized(error)),
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


def _read_payload(
    payload: bytes, spec: GridSpec, names: tuple[str, ...]
) -> dict[str, npt.NDArray[np.float32]]:
    with rasterio.open(io.BytesIO(payload)) as source:
        if source.count != len(names):
            raise ValueError(f"Expected {len(names)} bands from Earth Engine, got {source.count}")
        if (source.width, source.height) != (spec.n_cols, spec.n_rows):
            raise ValueError(
                f"Earth Engine returned {source.width}x{source.height}, "
                f"expected the grid's {spec.n_cols}x{spec.n_rows}"
            )
        return dict(zip(names, source.read().astype("float32"), strict=True))


def _write_cache(
    bands: dict[str, npt.NDArray[np.float32]],
    spec: GridSpec,
    names: tuple[str, ...],
    path: Path,
) -> None:
    data = np.stack([bands[name] for name in names])
    transform = rasterio.Affine(*grid_transform(spec))

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tif.partial")
    with rasterio.open(
        partial,
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
    partial.replace(path)


def cached_months(config: Config) -> dict[date, Path]:
    """Every month already on disk, read from the rasters' own band descriptions."""
    found: dict[date, Path] = {}
    for path in sorted(config.path(config.paths.landsat_raw).glob("*.tif")):
        with rasterio.open(path) as source:
            for description in source.descriptions:
                _, month = parse_band_name(description)
                found[month] = path
    return found


def fetch_landsat(config: Config, years: list[int], force: bool = False) -> list[Path]:
    """Download every chunk of the requested years, skipping what is already cached."""
    import ee

    from tfire.grid import load_grid

    spec, _ = load_grid(config)
    ee.Initialize(project=config.sources.gee_project)

    cached = set(cached_months(config))
    wanted = [chunk for chunk in chunk_starts(config) if chunk.year in set(years)]
    targets = [chunk_cache_path(config, chunk) for chunk in wanted]
    queue = [chunk for chunk in wanted if force or not set(chunk_months(config, chunk)) <= cached]
    if not queue:
        logger.info("All %d chunk(s) already cached", len(targets))
        return targets

    logger.info(
        "Fetching %d chunk(s) of %d month(s) over %d x %d cells; each composites every scene "
        "at %d m server side",
        len(queue),
        config.vegetation.chunk_months,
        spec.n_cols,
        spec.n_rows,
        _NATIVE_SCALE_M,
    )
    for index, chunk in enumerate(queue, start=1):
        months = [month for month in chunk_months(config, chunk) if force or month not in cached]
        try:
            written = [_fetch_window(config, spec, months)]
        except Exception as error:
            if not is_oversized(error):
                raise
            # scene density varies across the record: two OLI satellites plus residual ETM+
            # put a month well above what one request holds, where a 1980s TM month is trivial
            logger.warning(
                "Earth Engine refused %s as too heavy; retrying its %d month(s) one at a time",
                f"{chunk:%Y-%m}",
                len(months),
            )
            written = [_fetch_window(config, spec, [month]) for month in months]

        for path in written:
            logger.info(
                "[%d/%d] %s -> %s (%.1f MB)",
                index,
                len(queue),
                f"{chunk:%Y-%m}",
                path.name,
                path.stat().st_size / 1e6,
            )
    return targets


def chunk_of(config: Config, month: date) -> date:
    """The chunk a month belongs to, on the same lattice `chunk_starts` walks."""
    size = config.vegetation.chunk_months
    return date(month.year, ((month.month - 1) // size) * size + 1, 1)


def fetch_months(config: Config, months: list[date], force: bool = False) -> list[Path]:
    """Download whole chunks around the requested months, skipping what is already cached."""
    import ee

    from tfire.grid import load_grid

    cached = set(cached_months(config))
    wanted = sorted({chunk_of(config, month) for month in months})
    queue = [
        chunk
        for chunk in wanted
        if force or not {month for month in _chunk_span(config, chunk)} <= cached
    ]
    if not queue:
        logger.info("Every month requested is already cached")
        return [chunk_cache_path(config, chunk) for chunk in wanted]

    spec, _ = load_grid(config)
    ee.Initialize(project=config.sources.gee_project)

    written = []
    for chunk in queue:
        pending = [month for month in _chunk_span(config, chunk) if force or month not in cached]
        logger.info("Fetching %s: %d month(s)", f"{chunk:%Y-%m}", len(pending))
        written.append(_fetch_window(config, spec, pending))
    return written


def _chunk_span(config: Config, chunk: date) -> list[date]:
    """The months of one chunk, unclipped: `chunk_months` stops at the configured record."""
    size = config.vegetation.chunk_months
    months = []
    for offset in range(size):
        step = chunk.month + offset
        months.append(date(chunk.year + (step - 1) // 12, (step - 1) % 12 + 1, 1))
    return months


def _fetch_window(config: Config, spec: GridSpec, months: list[date]) -> Path:
    """Download one request's worth of months into a raster named for its first month.

    Falls back to one request per band group if the whole set is too heavy, which happens from
    2022 on where two OLI satellites and residual ETM+ overlap.
    """
    path = chunk_cache_path(config, months[0])
    names = tuple(band_name(name, month) for month in months for name in OUTPUT_BANDS)
    try:
        groups: tuple[tuple[str, ...], ...] = (OUTPUT_BANDS,)
        bands = _download_groups(config, spec, months, groups)
    except Exception as error:
        if not is_oversized(error) or len(months) > 1:
            raise
        logger.warning(
            "Earth Engine refused %s even as a single month; splitting its bands across "
            "%d requests",
            f"{months[0]:%Y-%m}",
            len(_BAND_GROUPS),
        )
        bands = _download_groups(config, spec, months, _BAND_GROUPS)

    _write_cache(bands, spec, names, path)
    return path


def _download_groups(
    config: Config,
    spec: GridSpec,
    months: list[date],
    groups: tuple[tuple[str, ...], ...],
) -> dict[str, npt.NDArray[np.float32]]:
    bands: dict[str, npt.NDArray[np.float32]] = {}
    for group in groups:
        names = tuple(band_name(name, month) for month in months for name in group)
        payload = _download(window_image(config, spec, months, group), spec)
        bands.update(_read_payload(payload, spec, names))
    return bands
