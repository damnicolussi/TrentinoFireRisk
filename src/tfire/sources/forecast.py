"""Open-Meteo hourly fields on the ERA5-Land backbone lattice: the archive and the forecast."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import numpy.typing as npt
import requests
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tfire.config import Config
from tfire.features.meteo import HOURLY_FIELDS, HourlyFields

if TYPE_CHECKING:
    from tfire.sources.era5land import Lattice

logger = logging.getLogger(__name__)

Provider = Literal["archive", "forecast"]

PROVIDERS: Final[tuple[Provider, ...]] = ("archive", "forecast")

# Open-Meteo name -> the field of `HourlyFields` it fills. Requested in this order, and the
# hourly block comes back keyed by these names.
FIELD_BY_VARIABLE: Final[dict[str, str]] = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpoint_c",
    "surface_pressure": "pres_hpa",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "soil_moisture_0_to_7cm": "soil_water_l1",
    "soil_moisture_7_to_28cm": "soil_water_l2",
}

# how far a returned point may sit from the one asked for. Every model snaps to its own grid,
# and IFS at 0.25 degrees snaps several backbone cells onto one, so this only has to be tight
# enough to catch a response that came back in a different order than it was asked for.
_SNAP_TOLERANCE_DEG: Final = 0.5

_RETRY_WAIT_MULTIPLIER_S: Final = 2
_RETRY_WAIT_MAX_S: Final = 75


class ForecastError(RuntimeError):
    """Open-Meteo refused the request or sent something the parser cannot use."""


class _RetryableError(ForecastError):
    """A transport failure or a 5xx, worth sending again."""


@dataclass(frozen=True)
class Endpoint:
    provider: Provider
    url: str
    model: str


@dataclass(frozen=True)
class Segment:
    """Which endpoint answered for which stretch of days, for the run's provenance."""

    provider: Provider
    start: date
    end: date


def endpoint(config: Config, provider: Provider) -> Endpoint:
    if provider == "archive":
        return Endpoint(provider, config.forecast.archive_url, config.forecast.archive_model)
    return Endpoint(provider, config.forecast.forecast_url, config.forecast.forecast_model)


def window(config: Config, provider: Provider, today: date) -> tuple[date, date]:
    """The span a provider can serve, as of `today`.

    Both windows slide with the calendar, and `ForecastConfig` refuses a setting that would
    leave a gap between them.
    """
    settings = config.forecast
    if provider == "archive":
        return date(1950, 1, 1), today - timedelta(days=settings.archive_latency_days)
    return (
        today - timedelta(days=settings.forecast_past_days),
        today + timedelta(days=settings.horizon_days),
    )


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _month_end(day: date) -> date:
    following = _month_start(day) + timedelta(days=32)
    return _month_start(following) - timedelta(days=1)


def spinup_window(config: Config, first: date, last: date, today: date) -> tuple[date, date]:
    """The span to actually download so that `[first, last]` has its spin-up."""
    start = _month_start(first - timedelta(days=config.forecast.spinup_days))
    horizon = today + timedelta(days=config.forecast.horizon_days)
    return start, min(_month_end(last), horizon)


def _cache_path(
    config: Config, spot: Endpoint, start: date, end: date, digest: str, today: date
) -> Path:
    # a forecast for the same span changes with every model run, so the day it was issued is
    # part of its identity; the archive's answer for a past span does not move
    issued = f"_{today:%Y%m%d}" if spot.provider == "forecast" else ""
    name = f"{start:%Y%m%d}-{end:%Y%m%d}_{digest}{issued}.npz"
    return config.path(config.paths.forecast_cache) / spot.provider / name


def _query(
    spot: Endpoint,
    latitudes: npt.NDArray[np.float64],
    longitudes: npt.NDArray[np.float64],
    start: date,
    end: date,
) -> dict[str, str]:
    return {
        "latitude": ",".join(f"{value:.4f}" for value in latitudes),
        "longitude": ",".join(f"{value:.4f}" for value in longitudes),
        # one per coordinate, which is how the API takes it. Without this Open-Meteo
        # statistically downscales temperature and pressure onto its own 90 m terrain, and the
        # backbone this has to match reads the model cell itself, orography and all.
        "elevation": ",".join(["nan"] * latitudes.size),
        "start_date": f"{start:%Y-%m-%d}",
        "end_date": f"{end:%Y-%m-%d}",
        "hourly": ",".join(FIELD_BY_VARIABLE),
        "models": spot.model,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }


def _request_digest(query: dict[str, str]) -> str:
    """Fingerprint of everything but the dates, so a changed request cannot read a stale cache."""
    stable = sorted((key, value) for key, value in query.items() if "date" not in key)
    payload = "&".join(f"{key}={value}" for key, value in stable)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _get(spot: Endpoint, params: dict[str, str], config: Config) -> list[dict[str, Any]]:
    try:
        response = requests.get(spot.url, params=params, timeout=config.forecast.request_timeout_s)
    except requests.RequestException as error:
        raise _RetryableError(f"{spot.url} unreachable: {error}") from error

    # 429 is Open-Meteo's own minutely quota, which clears on its own
    if response.status_code >= 500 or response.status_code == 429:
        raise _RetryableError(f"{spot.url} returned {response.status_code}")
    if not response.ok:
        # Open-Meteo puts the actual complaint in the body, which is far more useful than the code
        detail = response.json().get("reason", response.text[:200]) if response.content else ""
        raise ForecastError(f"{spot.url} returned {response.status_code}: {detail}")

    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def _fetch_batch(
    spot: Endpoint,
    latitudes: npt.NDArray[np.float64],
    longitudes: npt.NDArray[np.float64],
    start: date,
    end: date,
    config: Config,
) -> list[dict[str, Any]]:
    params = _query(spot, latitudes, longitudes, start, end)

    retrying = Retrying(
        stop=stop_after_attempt(config.forecast.retry_attempts),
        wait=wait_exponential(multiplier=_RETRY_WAIT_MULTIPLIER_S, max=_RETRY_WAIT_MAX_S),
        retry=retry_if_exception_type(_RetryableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    locations = retrying(_get, spot, params, config)

    if len(locations) != latitudes.size:
        raise ForecastError(
            f"Asked {spot.url} for {latitudes.size} location(s), got {len(locations)}"
        )
    return locations


def _snap_distance(
    locations: list[dict[str, Any]],
    latitudes: npt.NDArray[np.float64],
    longitudes: npt.NDArray[np.float64],
) -> float:
    returned_lat = np.array([item["latitude"] for item in locations], dtype="float64")
    returned_lon = np.array([item["longitude"] for item in locations], dtype="float64")
    return float(np.max(np.hypot(returned_lat - latitudes, returned_lon - longitudes)))


def _stack(locations: list[dict[str, Any]]) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """The hourly block of every location as `(n_hours, n_cells)` arrays, on one shared clock."""
    stamps = locations[0]["hourly"]["time"]
    times = np.array(stamps, dtype="datetime64[h]")

    columns: dict[str, Any] = {}
    for variable, field in FIELD_BY_VARIABLE.items():
        series = []
        for index, item in enumerate(locations):
            hourly = item["hourly"]
            if hourly["time"] != stamps:
                raise ForecastError(f"Location {index} came back on a different hourly clock")
            if variable not in hourly:
                raise ForecastError(f"Location {index} carries no {variable}")
            series.append(np.array(hourly[variable], dtype="float64"))
        columns[field] = np.stack(series, axis=1)

    return times, columns


def _read_cache(path: Path) -> HourlyFields:
    with np.load(path) as stored:
        return HourlyFields(
            times=stored["times"].astype("datetime64[h]"),
            **{field: stored[field] for field in FIELD_BY_VARIABLE.values()},
        )


def fetch_hourly(
    config: Config,
    lattice: Lattice,
    start: date,
    end: date,
    provider: Provider,
    today: date | None = None,
) -> HourlyFields:
    """Hourly fields for `[start, end]` at every backbone cell center, cached on disk.

    Cells come back in `era5_id` order, which is what makes `era5_weights.parquet` and the
    nearest-backbone lookup usable against the result without any remapping.
    """
    spot = endpoint(config, provider)
    latitudes = lattice.cell_latitudes()
    longitudes = lattice.cell_longitudes()

    digest = _request_digest(_query(spot, latitudes, longitudes, start, end))
    path = _cache_path(config, spot, start, end, digest, today or date.today())
    if path.is_file():
        logger.info("Reusing the cached %s response at %s", provider, path.name)
        return _read_cache(path)

    size = config.forecast.batch_size
    batches = [slice(first, first + size) for first in range(0, lattice.n_cells, size)]
    logger.info(
        "Fetching %s (%s) for %s to %s over %d cell(s) in %d request(s)",
        provider,
        spot.model,
        start,
        end,
        lattice.n_cells,
        len(batches),
    )

    locations: list[dict[str, Any]] = []
    snap = 0.0
    for batch in batches:
        part = _fetch_batch(spot, latitudes[batch], longitudes[batch], start, end, config)
        snap = max(snap, _snap_distance(part, latitudes[batch], longitudes[batch]))
        locations.extend(part)

    if snap > _SNAP_TOLERANCE_DEG:
        raise ForecastError(
            f"A returned point sits {snap:.3f} degrees from the one asked for, past the "
            f"{_SNAP_TOLERANCE_DEG} tolerance; the response may not be in request order"
        )

    times, columns = _stack(locations)
    logger.info(
        "%s: %d hour(s) x %d cell(s), largest grid snap %.3f degrees",
        provider,
        times.size,
        lattice.n_cells,
        snap,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=times.astype("int64"), **columns)
    return HourlyFields(times=times, **columns)


def plan_span(config: Config, start: date, end: date, today: date) -> list[Segment]:
    """Split `[start, end]` across the endpoints that can answer for it.

    The archive is preferred wherever it reaches: it is reanalysis of what happened, against a
    forecast of what might. Only the days past its latency fall to the forecast endpoint.
    """
    if end > today + timedelta(days=config.forecast.horizon_days):
        raise ForecastError(
            f"{end} is past the {config.forecast.horizon_days}-day forecast horizon from {today}"
        )

    archive_end = min(end, window(config, "archive", today)[1])
    forecast_start = max(start, archive_end + timedelta(days=1))
    earliest_forecast = window(config, "forecast", today)[0]

    segments = []
    if archive_end >= start:
        segments.append(Segment("archive", start, archive_end))
    if forecast_start <= end:
        if forecast_start < earliest_forecast:
            raise ForecastError(
                f"{forecast_start} is before the forecast endpoint's own history "
                f"({earliest_forecast}) and past the archive's latency; no endpoint covers it"
            )
        segments.append(Segment("forecast", forecast_start, end))
    return segments


def fetch_span(
    config: Config, lattice: Lattice, start: date, end: date, today: date | None = None
) -> tuple[HourlyFields, list[Segment]]:
    """One continuous hourly series over `[start, end]`, stitched across the endpoints.

    A span reaching from the past into the forecast crosses a seam where the model changes.
    The FWI codes carry that seam forward, which is the price of running them on a forecast at
    all; the segments are returned so a caller can record where it fell.
    """
    stamp = today or date.today()
    segments = plan_span(config, start, end, stamp)

    parts = [
        fetch_hourly(config, lattice, piece.start, piece.end, piece.provider, stamp)
        for piece in segments
    ]
    if len(parts) == 1:
        return parts[0], segments

    times = np.concatenate([part.times for part in parts])
    steps = np.unique(np.diff(times.astype("int64")))
    if steps.size != 1 or steps[0] != 1:
        raise ForecastError(f"The stitched series is not hourly and contiguous: steps {steps}")

    stitched = HourlyFields(
        times=times,
        **{
            field: np.concatenate([getattr(part, field) for part in parts])
            for field in HOURLY_FIELDS
        },
    )
    logger.info(
        "Stitched %s into %d hour(s)",
        " then ".join(f"{piece.provider} {piece.start} to {piece.end}" for piece in segments),
        times.size,
    )
    return stitched, segments
