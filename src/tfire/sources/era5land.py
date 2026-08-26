"""ERA5-Land hourly fields from Earth Engine, cached per variable and half-year."""

from __future__ import annotations

import io
import logging
import math
import time
from calendar import monthrange
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from tfire.config import Config
from tfire.sources.landsat import is_oversized

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# what each variable is called inside the cached NetCDF
SHORT_NAMES: Final[dict[str, str]] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
}

RESOLUTION_DEG: Final = 0.1

GEE_COLLECTION: Final = "ECMWF/ERA5_LAND/HOURLY"

# Earth Engine's spelling of the same fields
GEE_BANDS: Final[dict[str, str]] = {
    "2m_temperature": "temperature_2m",
    "2m_dewpoint_temperature": "dewpoint_temperature_2m",
    "surface_pressure": "surface_pressure",
    "total_precipitation": "total_precipitation",
    "10m_u_component_of_wind": "u_component_of_wind_10m",
    "10m_v_component_of_wind": "v_component_of_wind_10m",
    "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2": "volumetric_soil_water_layer_2",
}

# `getDownloadURL` refuses more bands than this, and one window carries hours x variables
MAX_GEE_BANDS: Final = 1024

# ERA5-Land covers Trentino with no masked cell, and Earth Engine writes a masked pixel as 0,
# which would read as a real value. A sentinel no field can reach turns that into a loud failure.
_MASK_FILL: Final = -32768.0

_GEE_TIMEOUT_S: Final = 1800
_GEE_RETRY_ATTEMPTS: Final = 4

# a bounding box edge sitting on a grid line lands just under it in binary
_GRID_EPS: Final = 1e-6

HALVES: Final = (1, 2)


@dataclass(frozen=True, eq=False)
class Lattice:
    """The 0.1 degree ERA5-Land axes cropped to the request area.

    `era5_id` runs row-major from the north-west, the same convention `GridSpec` uses for
    `cell_id`, so the two indexing schemes cannot drift apart.
    """

    latitudes: npt.NDArray[np.float64]
    longitudes: npt.NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lattice):
            return NotImplemented
        return np.array_equal(self.latitudes, other.latitudes) and np.array_equal(
            self.longitudes, other.longitudes
        )

    def __post_init__(self) -> None:
        if np.any(np.diff(self.latitudes) >= 0):
            raise ValueError("Latitudes must run north to south")
        if np.any(np.diff(self.longitudes) <= 0):
            raise ValueError("Longitudes must run west to east")

    @property
    def n_rows(self) -> int:
        return int(self.latitudes.size)

    @property
    def n_cols(self) -> int:
        return int(self.longitudes.size)

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    def cell_latitudes(self) -> npt.NDArray[np.float64]:
        return np.repeat(self.latitudes, self.n_cols)

    def cell_longitudes(self) -> npt.NDArray[np.float64]:
        return np.tile(self.longitudes, self.n_rows)


def fetch_end(config: Config) -> date:
    """Last date the backbone is meant to cover, extension included."""
    return config.meteo.extension_end or config.date_range.end


def fetch_halves(config: Config) -> list[tuple[int, int]]:
    """Half-years the backbone covers, whole ones only, spin-up included."""
    first = config.date_range.start.year - config.meteo.spinup_years
    end = fetch_end(config)
    return [
        (year, half)
        for year in range(first, end.year + 1)
        for half in HALVES
        if half_end(year, half) <= end
    ]


def fetch_years(config: Config) -> list[int]:
    """Calendar years with at least one whole half-year in the record."""
    return sorted({year for year, _ in fetch_halves(config)})


def year_halves(config: Config, year: int) -> list[int]:
    return [half for cached_year, half in fetch_halves(config) if cached_year == year]


def half_months(half: int) -> list[int]:
    """The calendar months covered by one half of a year."""
    if half not in HALVES:
        raise ValueError(f"A year has halves {list(HALVES)}, not {half}")
    return list(range(1, 7) if half == 1 else range(7, 13))


def half_end(year: int, half: int) -> date:
    month = half_months(half)[-1]
    return date(year, month, monthrange(year, month)[1])


def cache_path(config: Config, variable: str, year: int, half: int) -> Path:
    return config.path(config.paths.era5_raw) / variable / f"{year}-h{half}.nc"


def bbox_lattice(config: Config) -> Lattice:
    """The 0.1 degree ERA5-Land points inside the request box, without reading a download."""
    box = config.sources.bbox_wgs84
    north = math.floor(box.north / RESOLUTION_DEG + _GRID_EPS)
    south = math.ceil(box.south / RESOLUTION_DEG - _GRID_EPS)
    west = math.ceil(box.west / RESOLUTION_DEG - _GRID_EPS)
    east = math.floor(box.east / RESOLUTION_DEG + _GRID_EPS)
    if north < south or east < west:
        raise ValueError(f"No ERA5-Land grid point inside {box}")

    latitudes = np.round(np.arange(north, south - 1, -1) * RESOLUTION_DEG, 6)
    longitudes = np.round(np.arange(west, east + 1) * RESOLUTION_DEG, 6)
    return Lattice(latitudes, longitudes)


def half_bounds(year: int, half: int) -> tuple[datetime, datetime]:
    """The half-open UTC interval one half-year covers."""
    months = half_months(half)
    start = datetime(year, months[0], 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC) if half == 2 else datetime(year, 7, 1, tzinfo=UTC)
    return start, end


def half_hours(year: int, half: int) -> int:
    return 24 * sum(monthrange(year, month)[1] for month in half_months(half))


def windows(config: Config, year: int, half: int) -> list[tuple[datetime, datetime]]:
    """Half-open intervals tiling one half-year, each within the band cap."""
    span = config.meteo.gee_window_hours
    bands = span * len(config.meteo.variables)
    if bands > MAX_GEE_BANDS:
        raise ValueError(
            f"{span} hours x {len(config.meteo.variables)} variables is {bands} bands, "
            f"over Earth Engine's limit of {MAX_GEE_BANDS}; lower meteo.gee_window_hours"
        )

    start, end = half_bounds(year, half)
    step = timedelta(hours=span)
    spans = []
    cursor = start
    while cursor < end:
        spans.append((cursor, min(cursor + step, end)))
        cursor += step
    return spans


@retry(
    stop=stop_after_attempt(_GEE_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=5, max=120),
    retry=retry_if_exception(lambda error: not is_oversized(error)),
    reraise=True,
)
def _download_window(
    config: Config, lattice: Lattice, start: datetime, end: datetime
) -> tuple[list[str], bytes]:
    import ee

    bands = [GEE_BANDS[name] for name in config.meteo.variables]
    collection = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterDate(start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))
        .select(bands)
    )
    stamps: list[str] = collection.aggregate_array("system:index").getInfo()

    west = float(lattice.longitudes[0]) - RESOLUTION_DEG / 2
    north = float(lattice.latitudes[0]) + RESOLUTION_DEG / 2
    east = float(lattice.longitudes[-1]) + RESOLUTION_DEG / 2
    south = float(lattice.latitudes[-1]) - RESOLUTION_DEG / 2
    region = ee.Geometry.Rectangle(
        [west, south, east, north], proj=ee.Projection("EPSG:4326"), geodesic=False
    )
    url = (
        collection.toBands()
        .unmask(_MASK_FILL)
        .getDownloadURL(
            {
                "region": region,
                "crs": "EPSG:4326",
                "crs_transform": [RESOLUTION_DEG, 0, west, 0, -RESOLUTION_DEG, north],
                "format": "GEO_TIFF",
            }
        )
    )
    response = requests.get(url, timeout=_GEE_TIMEOUT_S)
    response.raise_for_status()
    return stamps, response.content


def crop_indices(
    transform: rasterio.Affine, width: int, height: int, lattice: Lattice
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Rows and columns of the lattice points inside a raster Earth Engine may have grown.

    The returned extent is snapped outward to whole tiles, so it is generally larger than the
    box asked for and its origin is not the one asked for either.
    """
    longitudes = transform.c + transform.a * (np.arange(width) + 0.5)
    latitudes = transform.f + transform.e * (np.arange(height) + 0.5)

    rows = np.abs(latitudes[None, :] - lattice.latitudes[:, None]).argmin(axis=1)
    columns = np.abs(longitudes[None, :] - lattice.longitudes[:, None]).argmin(axis=1)
    if not np.allclose(latitudes[rows], lattice.latitudes, atol=_GRID_EPS):
        raise ValueError(f"Raster latitudes {latitudes} do not carry {lattice.latitudes}")
    if not np.allclose(longitudes[columns], lattice.longitudes, atol=_GRID_EPS):
        raise ValueError(f"Raster longitudes {longitudes} do not carry {lattice.longitudes}")
    return rows, columns


@contextmanager
def _quiet_gdal() -> Iterator[None]:
    """Earth Engine's band stack carries no photometric tag, and GDAL warns about it per file."""
    gdal = logging.getLogger("rasterio._env")
    level = gdal.level
    gdal.setLevel(logging.ERROR)
    try:
        yield
    finally:
        gdal.setLevel(level)


def read_window(
    payload: bytes, lattice: Lattice, n_hours: int, n_variables: int
) -> npt.NDArray[np.float32]:
    """One window's GeoTIFF as `(hour, variable, latitude, longitude)` on the lattice.

    `toBands` emits one band per image per selected band, images outermost, so the band
    holding variable `v` at hour `h` sits at `h * n_variables + v`.
    """
    with _quiet_gdal(), rasterio.open(io.BytesIO(payload)) as source:
        expected = n_hours * n_variables
        if source.count != expected:
            raise ValueError(
                f"Earth Engine returned {source.count} bands, expected "
                f"{n_hours} hours x {n_variables} variables = {expected}"
            )
        rows, columns = crop_indices(source.transform, source.width, source.height, lattice)
        stack = source.read().astype("float32")

    cropped: npt.NDArray[np.float32] = stack[:, rows[:, None], columns[None, :]]
    if np.any(cropped == _MASK_FILL):
        raise ValueError("Earth Engine masked an ERA5-Land cell over Trentino")
    return cropped.reshape(n_hours, n_variables, lattice.n_rows, lattice.n_cols)


def _write_half(
    config: Config,
    lattice: Lattice,
    times: npt.NDArray[np.datetime64],
    values: npt.NDArray[np.float32],
    year: int,
    half: int,
) -> list[Path]:
    import xarray as xr

    written = []
    for index, variable in enumerate(config.meteo.variables):
        short = SHORT_NAMES[variable]
        dataset = xr.Dataset(
            {short: (("valid_time", "latitude", "longitude"), values[:, index])},
            coords={
                "valid_time": times,
                "latitude": lattice.latitudes,
                "longitude": lattice.longitudes,
            },
        )
        target = cache_path(config, variable, year, half)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".nc.partial")
        dataset.to_netcdf(partial, encoding={short: {"zlib": True, "complevel": 4}})
        partial.replace(target)
        written.append(target)
    return written


def _fetch_half(config: Config, lattice: Lattice, year: int, half: int) -> list[Path]:
    spans = windows(config, year, half)
    with ThreadPoolExecutor(max_workers=config.meteo.gee_workers) as pool:
        results = list(pool.map(lambda span: _download_window(config, lattice, *span), spans))

    stamps: list[str] = []
    blocks = []
    for window_stamps, payload in results:
        stamps.extend(window_stamps)
        blocks.append(
            read_window(payload, lattice, len(window_stamps), len(config.meteo.variables))
        )

    expected = half_hours(year, half)
    if len(stamps) != expected:
        raise ValueError(f"{year} h{half} came back with {len(stamps)} hours, expected {expected}")

    times = np.array(
        [datetime.strptime(stamp, "%Y%m%dT%H") for stamp in stamps], dtype="datetime64[ns]"
    )
    if np.any(np.diff(times) != np.timedelta64(1, "h")):
        raise ValueError(f"{year} h{half} is not a contiguous hourly series")

    return _write_half(config, lattice, times, np.concatenate(blocks), year, half)


def fetch_era5(config: Config, years: list[int], force: bool = False) -> list[Path]:
    """Download every variable and half-year, skipping what is already cached."""
    import ee

    unknown = sorted(set(config.meteo.variables) - set(GEE_BANDS))
    if unknown:
        raise ValueError(f"No Earth Engine band known for {unknown}; extend GEE_BANDS")

    wanted = [(year, half) for year, half in fetch_halves(config) if year in set(years)]
    targets = [
        cache_path(config, variable, year, half)
        for year, half in wanted
        for variable in config.meteo.variables
    ]
    queue = [(year, half) for year, half in wanted if force or not has_half(config, year, half)]
    if not queue:
        logger.info("All %d chunk(s) already cached", len(targets))
        return targets

    ee.Initialize(project=config.sources.gee_project)
    lattice = bbox_lattice(config)
    logger.info(
        "Fetching %d half-year(s) over %d x %d cells, %d hour windows, %d at a time",
        len(queue),
        lattice.n_rows,
        lattice.n_cols,
        config.meteo.gee_window_hours,
        config.meteo.gee_workers,
    )

    for index, (year, half) in enumerate(queue, start=1):
        started = time.monotonic()
        written = _fetch_half(config, lattice, year, half)
        size = sum(path.stat().st_size for path in written)
        logger.info(
            "[%d/%d] %d h%d -> %d file(s) (%.1f MB, %.0fs)",
            index,
            len(queue),
            year,
            half,
            len(written),
            size / 1e6,
            time.monotonic() - started,
        )
    return targets


def has_half(config: Config, year: int, half: int) -> bool:
    """Whether every variable is cached for one half-year."""
    return all(cache_path(config, name, year, half).is_file() for name in config.meteo.variables)


def missing_years(config: Config, years: list[int]) -> list[int]:
    """Years with at least one chunk not yet cached."""
    return [
        year
        for year in years
        if not all(has_half(config, year, half) for half in year_halves(config, year))
    ]


def open_half(config: Config, year: int, half: int) -> xr.Dataset:
    """Merge the cached variables for one half-year into a `(time, era5_id)` dataset.

    Flattens the lat/lon grid so every downstream step works on one series per backbone
    cell, and keeps latitude and longitude as coordinates along `era5_id`.
    """
    import xarray as xr

    parts = [_read_chunk(config, name, year, half) for name in config.meteo.variables]
    return _stack_cells(xr.merge(parts, join="exact"))


def open_year(config: Config, year: int) -> xr.Dataset:
    import xarray as xr

    halves = year_halves(config, year)
    if not halves:
        raise ValueError(f"{year} has no whole half-year inside the record")
    return xr.concat([open_half(config, year, half) for half in halves], dim="time")


def _read_chunk(config: Config, variable: str, year: int, half: int) -> xr.Dataset:
    """One cached chunk, read out in full rather than left as an open handle."""
    import xarray as xr

    path = cache_path(config, variable, year, half)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ERA5-Land cache, run fetch-era5 first: {path}")

    short = SHORT_NAMES[variable]
    with xr.open_dataset(path) as chunk:
        if short not in chunk.data_vars:
            raise ValueError(f"{path} holds {list(chunk.data_vars)}, expected {short!r}")
        return _normalize(chunk[[short]]).load()


def _normalize(dataset: xr.Dataset) -> xr.Dataset:
    """Give every download the same dimension names and axis directions."""
    if "valid_time" in dataset.dims:
        dataset = dataset.rename({"valid_time": "time"})

    if dataset["latitude"][0] < dataset["latitude"][-1]:
        dataset = dataset.isel(latitude=slice(None, None, -1))
    return dataset


def _stack_cells(dataset: xr.Dataset) -> xr.Dataset:
    stacked = dataset.stack(era5_id=("latitude", "longitude"), create_index=False)
    stacked = stacked.transpose("time", "era5_id")
    return stacked.assign_coords(era5_id=np.arange(stacked.sizes["era5_id"], dtype="int32"))


def read_lattice(config: Config) -> Lattice:
    """The backbone axes, from a cached download if there is one, otherwise from the request box."""
    import xarray as xr

    for year, half in fetch_halves(config):
        for variable in config.meteo.variables:
            path = cache_path(config, variable, year, half)
            if path.is_file():
                with xr.open_dataset(path) as dataset:
                    return lattice_of(_stack_cells(_normalize(dataset)))

    logger.info("No ERA5-Land download cached, taking the lattice from the request box")
    return bbox_lattice(config)


def lattice_of(dataset: xr.Dataset) -> Lattice:
    """Recover the 0.1 degree axes from a stacked dataset."""
    latitudes = np.unique(dataset["latitude"].to_numpy())[::-1]
    longitudes = np.unique(dataset["longitude"].to_numpy())
    lattice = Lattice(latitudes.astype(np.float64), longitudes.astype(np.float64))
    if lattice.n_cells != dataset.sizes["era5_id"]:
        raise ValueError(
            f"{dataset.sizes['era5_id']} cells do not form a "
            f"{lattice.n_rows}x{lattice.n_cols} lattice"
        )
    return lattice
