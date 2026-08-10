"""ERA5-Land hourly fields from the Copernicus CDS, cached per variable and half-year."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from tfire.config import Config

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# what each requested variable is called inside the returned NetCDF
SHORT_NAMES: Final[dict[str, str]] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}

RESOLUTION_DEG: Final = 0.1

# CDS rejects a request above roughly 6,000 fields. One variable over half a year is
# 4,464, and keeping a single variable per request avoids the archive the CDS returns
# when accumulated and instantaneous fields share one.
HALVES: Final = (1, 2)

# failures carrying one of these are about the request, not about load
_PERMANENT_MARKERS: Final = ("too large", "cost limit", "licence", "license", "not found")

# the queue is full right now, so the same request will succeed later
_THROTTLE_MARKERS: Final = ("temporarily limited", "has been rejected", "too many", "rate limit")

_MAX_COOLDOWN_S: Final = 600

# dimensions the CDS adds for the near-real-time overlap and for ensemble products
_SURPLUS_DIMS: Final = ("number", "expver")


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


def fetch_years(config: Config) -> range:
    """Calendar years to download, including the spin-up years before the modeling window."""
    start = config.date_range.start.year - config.meteo.spinup_years
    return range(start, config.date_range.end.year + 1)


def half_months(half: int) -> list[int]:
    """The calendar months covered by one half of a year."""
    if half not in HALVES:
        raise ValueError(f"A year has halves {list(HALVES)}, not {half}")
    return list(range(1, 7) if half == 1 else range(7, 13))


def cache_path(config: Config, variable: str, year: int, half: int) -> Path:
    return config.path(config.paths.era5_raw) / variable / f"{year}-h{half}.nc"


def _request(config: Config, variable: str, year: int, half: int) -> dict[str, object]:
    return {
        "variable": [variable],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in half_months(half)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": config.sources.bbox_wgs84.as_cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _reason(error: BaseException) -> str:
    """The error text together with whatever the API put in the response body."""
    response = getattr(error, "response", None)
    body = getattr(response, "text", "") if response is not None else ""
    return f"{error} {body}".lower()


def is_permanent(error: BaseException) -> bool:
    """Whether resubmitting the same request would fail the same way."""
    reason = _reason(error)
    return any(marker in reason for marker in _PERMANENT_MARKERS)


def is_throttle(error: BaseException) -> bool:
    """Whether the CDS turned the request away for load rather than for content."""
    reason = _reason(error)
    return not is_permanent(error) and any(marker in reason for marker in _THROTTLE_MARKERS)


@dataclass
class _Chunk:
    variable: str
    year: int
    half: int
    target: Path
    attempts: int = 0

    def __str__(self) -> str:
        return f"{self.variable} {self.year} h{self.half}"


def _pending(config: Config, years: list[int], force: bool) -> list[_Chunk]:
    chunks = []
    for year in years:
        for variable in config.meteo.variables:
            for half in HALVES:
                target = cache_path(config, variable, year, half)
                if target.is_file() and not force:
                    continue
                chunks.append(_Chunk(variable, year, half, target))
    return chunks


def _collect(remote: object, target: Path) -> None:
    """Download a finished job through a partial file."""
    partial = target.with_suffix(".nc.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    remote.download(str(partial))  # type: ignore[attr-defined]
    partial.replace(target)


def _defer(chunk: _Chunk, error: BaseException, config: Config, queue: list[_Chunk]) -> bool:
    """Requeue a chunk, and report whether the CDS was the one saying no."""
    if is_throttle(error):
        queue.append(chunk)
        return True

    chunk.attempts += 1
    if is_permanent(error) or chunk.attempts >= config.meteo.retry_attempts:
        raise error

    logger.warning("%s failed (attempt %d), requeued: %s", chunk, chunk.attempts, error)
    queue.append(chunk)
    return False


def fetch_era5(config: Config, years: list[int], force: bool = False) -> list[Path]:
    """Download every variable and half-year, skipping what is already cached."""
    import cdsapi

    unknown = sorted(set(config.meteo.variables) - set(SHORT_NAMES))
    if unknown:
        raise ValueError(f"No NetCDF short name known for {unknown}; extend SHORT_NAMES")

    targets = [
        cache_path(config, variable, year, half)
        for year in years
        for variable in config.meteo.variables
        for half in HALVES
    ]
    queue = _pending(config, years, force)
    if not queue:
        logger.info("All %d chunk(s) already cached", len(targets))
        return targets

    client = cdsapi.Client(
        timeout=config.meteo.request_timeout_s,
        progress=False,
        quiet=True,
        wait_until_complete=False,
    )
    logger.info("Fetching %d chunk(s), up to %d in flight", len(queue), config.meteo.max_in_flight)

    total = len(queue)
    written = 0
    cooldown_until = 0.0
    cooldown = float(config.meteo.poll_interval_s)
    in_flight: list[tuple[_Chunk, object]] = []

    while queue or in_flight:
        throttled = False
        while (
            queue
            and len(in_flight) < config.meteo.max_in_flight
            and time.monotonic() >= cooldown_until
        ):
            chunk = queue.pop(0)
            try:
                request = _request(config, chunk.variable, chunk.year, chunk.half)
                job = client.retrieve(config.meteo.dataset, request)
            except Exception as error:
                throttled |= _defer(chunk, error, config, queue)
                break
            in_flight.append((chunk, job))

        ready: list[tuple[_Chunk, object]] = []
        waiting: list[tuple[_Chunk, object]] = []
        for chunk, job in in_flight:
            try:
                (ready if job.results_ready else waiting).append((chunk, job))  # type: ignore[attr-defined]
            except Exception as error:
                throttled |= _defer(chunk, error, config, queue)
        in_flight = waiting

        for chunk, job in ready:
            _collect(job, chunk.target)
            written += 1
            logger.info(
                "[%d/%d] %s -> %s (%.1f MB)",
                written,
                total,
                chunk,
                chunk.target.name,
                chunk.target.stat().st_size / 1e6,
            )

        if throttled:
            logger.warning(
                "CDS queue is full, holding %d chunk(s) for %.0fs (%d in flight)",
                len(queue),
                cooldown,
                len(in_flight),
            )
            cooldown_until = time.monotonic() + cooldown
            cooldown = min(cooldown * 2, _MAX_COOLDOWN_S)
        elif ready:
            cooldown = float(config.meteo.poll_interval_s)

        if not ready and (in_flight or queue):
            time.sleep(config.meteo.poll_interval_s)

        if not ready and in_flight:
            time.sleep(config.meteo.poll_interval_s)

    return targets


def has_half(config: Config, year: int, half: int) -> bool:
    """Whether every variable is cached for one half-year."""
    return all(cache_path(config, name, year, half).is_file() for name in config.meteo.variables)


def missing_years(config: Config, years: list[int]) -> list[int]:
    """Years with at least one chunk not yet cached."""
    return [year for year in years if not all(has_half(config, year, half) for half in HALVES)]


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

    return xr.concat([open_half(config, year, half) for half in HALVES], dim="time")


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

    for dim in _SURPLUS_DIMS:
        if dim in dataset.dims:
            dataset = dataset.isel({dim: 0}, drop=True)
        elif dim in dataset.coords:
            dataset = dataset.drop_vars(dim)

    if dataset["latitude"][0] < dataset["latitude"][-1]:
        dataset = dataset.isel(latitude=slice(None, None, -1))
    return dataset


def _stack_cells(dataset: xr.Dataset) -> xr.Dataset:
    stacked = dataset.stack(era5_id=("latitude", "longitude"), create_index=False)
    stacked = stacked.transpose("time", "era5_id")
    return stacked.assign_coords(era5_id=np.arange(stacked.sizes["era5_id"], dtype="int32"))


def read_lattice(config: Config) -> Lattice:
    """The backbone axes, read from whichever download is already cached."""
    import xarray as xr

    for year in fetch_years(config):
        for variable in config.meteo.variables:
            for half in HALVES:
                path = cache_path(config, variable, year, half)
                if path.is_file():
                    with xr.open_dataset(path) as dataset:
                        return lattice_of(_stack_cells(_normalize(dataset)))

    raise FileNotFoundError("No ERA5-Land download cached; run `tfire fetch-era5` first")


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
