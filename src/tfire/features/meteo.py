"""Daily meteorological aggregates on the ERA5-Land backbone, plus the weights onto the grid."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import load_grid
from tfire.sources.era5land import (
    HALVES,
    Lattice,
    fetch_years,
    has_half,
    lattice_of,
    missing_years,
    open_half,
    open_year,
)

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

HOURS_PER_DAY: Final = 24

# fire weather is defined at noon local standard time
NOON_HOUR_LST: Final = 12

_KELVIN_OFFSET: Final = 273.15
_PA_TO_HPA: Final = 0.01
_M_TO_MM: Final = 1000.0

# Magnus coefficients over water
_MAGNUS_A: Final = 17.625
_MAGNUS_B: Final = 243.04

# tail of the previous year read alongside each year, so its first local day is whole
_OVERLAP_HOURS: Final = 48

_AGGREGATED: Final = ("temp", "rh", "pres")

HOURLY_FIELDS: Final = (
    "temp_c",
    "dewpoint_c",
    "pres_hpa",
    "precip_mm",
    "wind_speed",
    "wind_dir",
    "soil_water_l1",
    "soil_water_l2",
)

# volumetric soil water, m3/m3, taken as a daily mean only: it is a slow state variable, and a
# daily minimum or range of it is noise from the reanalysis rather than a diurnal signal
SOIL_LAYERS: Final = ("soil_water_l1", "soil_water_l2")

NOON_COLUMNS: Final = ("temp_noon", "rh_noon", "wind_speed_noon", "precip_noon24")

DIRECTIONS: Final = ("wind_dir_mean", "wind_dir_at_max")


def relative_humidity(
    temp_c: npt.NDArray[np.float64], dewpoint_c: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Magnus formula over water.

    Belongs at the hourly step: humidity is nonlinear in temperature and dewpoint, so
    feeding it daily means does not give the daily mean humidity.
    """
    saturation = _MAGNUS_A * temp_c / (_MAGNUS_B + temp_c)
    actual = _MAGNUS_A * dewpoint_c / (_MAGNUS_B + dewpoint_c)
    return np.clip(100.0 * np.exp(actual - saturation), 0.0, 100.0)


def wind_speed_direction(
    u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Magnitude and the meteorological direction the wind blows from, per hour.

    Averaging the components first collapses the magnitude toward zero on a day whose
    wind reverses, so this too runs before any daily aggregation.
    """
    speed = np.hypot(u, v)
    return speed, wrap_degrees(270.0 - np.degrees(np.arctan2(v, u)))


def wrap_degrees(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bearings in [0, 360). A tiny negative angle rounds up to exactly 360 under `%`."""
    wrapped = np.asarray(values) % 360.0
    return np.where(wrapped >= 360.0, 0.0, wrapped)


def circular_mean(degrees: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    """Mean bearing. An arithmetic mean of 350 and 10 points the wind due south."""
    radians = np.radians(degrees)
    mean = np.arctan2(np.sin(radians).mean(axis=axis), np.cos(radians).mean(axis=axis))
    return wrap_degrees(np.degrees(mean))


def deaccumulate(
    accumulated: npt.NDArray[np.float64], utc_hour: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Hourly amounts from ERA5-Land's running total since 00 UTC."""
    accumulated = np.asarray(accumulated, dtype=np.float64)
    amounts = np.empty_like(accumulated)
    amounts[1:] = accumulated[1:] - accumulated[:-1]
    amounts[0] = np.nan

    first = utc_hour == 1
    amounts[first] = accumulated[first]
    return np.maximum(amounts, 0.0)


def _whole_days(local_hour: npt.NDArray[np.int64]) -> tuple[int, int]:
    """Slice bounds covering only local days present in full."""
    starts = np.flatnonzero(local_hour == 0)
    ends = np.flatnonzero(local_hour == HOURS_PER_DAY - 1)
    if not starts.size or not ends.size or ends[-1] < starts[0]:
        raise ValueError("The hourly series does not cover one complete local day")

    start, stop = int(starts[0]), int(ends[-1]) + 1
    if (stop - start) % HOURS_PER_DAY:
        raise ValueError(f"{stop - start} hours between local midnights is not whole days")
    return start, stop


def _noon_accumulation(hourly: npt.NDArray[np.float64], n_days: int) -> npt.NDArray[np.float64]:
    """Rain over the 24 hours ending at noon, which is what the FWI codes consume."""
    padded = np.concatenate([np.zeros((1, hourly.shape[1])), np.cumsum(hourly, axis=0)])
    noon = np.arange(n_days) * HOURS_PER_DAY + NOON_HOUR_LST
    total: npt.NDArray[np.float64] = padded[noon + 1] - padded[noon + 1 - HOURS_PER_DAY]

    # the first day's window reaches before the series
    total[0] = np.nan
    return total


@dataclass(frozen=True)
class HourlyFields:
    """One hourly span in the units the daily aggregation works in, shaped `(n_hours, n_cells)`."""

    times: npt.NDArray[Any]
    temp_c: npt.NDArray[np.float64]
    dewpoint_c: npt.NDArray[np.float64]
    pres_hpa: npt.NDArray[np.float64]
    precip_mm: npt.NDArray[np.float64]
    wind_speed: npt.NDArray[np.float64]
    wind_dir: npt.NDArray[np.float64]
    soil_water_l1: npt.NDArray[np.float64]
    soil_water_l2: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        shapes = {name: getattr(self, name).shape for name in HOURLY_FIELDS}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Hourly fields disagree on shape: {shapes}")
        if self.times.shape[0] != self.temp_c.shape[0]:
            raise ValueError(
                f"{self.times.shape[0]} timestamps for {self.temp_c.shape[0]} hourly rows"
            )


def aggregate_daily(
    fields: HourlyFields, config: Config
) -> tuple[npt.NDArray[Any], dict[str, npt.NDArray[np.float64]]]:
    """Aggregate one hourly span into `(n_days, n_cells)` arrays, one per feature."""
    local = fields.times + np.timedelta64(config.meteo.utc_offset_hours, "h")
    start, stop = _whole_days(local.astype("int64") % HOURS_PER_DAY)
    n_days = (stop - start) // HOURS_PER_DAY

    def block(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return values[start:stop].reshape(n_days, HOURS_PER_DAY, -1)

    blocks = {
        "temp": block(fields.temp_c),
        "rh": block(relative_humidity(fields.temp_c, fields.dewpoint_c)),
        "pres": block(fields.pres_hpa),
    }
    precip_block = block(fields.precip_mm)
    speed_block = block(fields.wind_speed)
    direction_block = block(fields.wind_dir)

    columns: dict[str, npt.NDArray[np.float64]] = {}
    for name in _AGGREGATED:
        values = blocks[name]
        columns[f"{name}_mean"] = values.mean(axis=1)
        columns[f"{name}_min"] = values.min(axis=1)
        columns[f"{name}_max"] = values.max(axis=1)
        columns[f"{name}_range"] = columns[f"{name}_max"] - columns[f"{name}_min"]

    columns["precip_sum"] = precip_block.sum(axis=1)
    columns["wind_speed_mean"] = speed_block.mean(axis=1)
    columns["wind_speed_max"] = speed_block.max(axis=1)
    columns["wind_dir_mean"] = circular_mean(direction_block, axis=1)

    peak = speed_block.argmax(axis=1)
    columns["wind_dir_at_max"] = np.take_along_axis(
        direction_block, peak[:, None, :], axis=1
    ).squeeze(1)

    for layer in SOIL_LAYERS:
        columns[f"{layer}_mean"] = block(getattr(fields, layer)).mean(axis=1)

    columns["temp_noon"] = blocks["temp"][:, NOON_HOUR_LST]
    columns["rh_noon"] = blocks["rh"][:, NOON_HOUR_LST]
    columns["wind_speed_noon"] = speed_block[:, NOON_HOUR_LST]
    columns["precip_noon24"] = _noon_accumulation(fields.precip_mm[start:stop], n_days)

    dates = local[start:stop:HOURS_PER_DAY].astype("datetime64[D]")
    return dates, columns


def era5_hourly(dataset: xr.Dataset) -> HourlyFields:
    """ERA5-Land's stored fields in the units `aggregate_daily` works in."""
    times = dataset["time"].to_numpy().astype("datetime64[h]")
    utc_hour = times.astype("int64") % HOURS_PER_DAY

    speed, direction = wind_speed_direction(dataset["u10"].to_numpy(), dataset["v10"].to_numpy())

    # the opening stamp carries an accumulation that began before the series unless it is 01 UTC
    usable = 0 if utc_hour[0] == 1 else 1
    return HourlyFields(
        times=times[usable:],
        temp_c=dataset["t2m"].to_numpy()[usable:] - _KELVIN_OFFSET,
        dewpoint_c=dataset["d2m"].to_numpy()[usable:] - _KELVIN_OFFSET,
        pres_hpa=dataset["sp"].to_numpy()[usable:] * _PA_TO_HPA,
        precip_mm=deaccumulate(dataset["tp"].to_numpy() * _M_TO_MM, utc_hour)[usable:],
        wind_speed=speed[usable:],
        wind_dir=direction[usable:],
        soil_water_l1=dataset["swvl1"].to_numpy()[usable:],
        soil_water_l2=dataset["swvl2"].to_numpy()[usable:],
    )


def hourly_span(config: Config, year: int) -> xr.Dataset:
    """One year of hourly fields, preceded by enough of the previous year to close its first day."""
    import xarray as xr

    current = open_year(config, year)
    if not has_half(config, year - 1, HALVES[-1]):
        return current

    tail = open_half(config, year - 1, HALVES[-1]).isel(time=slice(-_OVERLAP_HOURS, None))
    return xr.concat([tail, current], dim="time")


def trailing_sum(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling sum over the window ending on, and including, each day."""
    if window < 1:
        raise ValueError(f"Window must be at least one day, got {window}")

    padded = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    total = np.full(values.shape, np.nan)
    total[window - 1 :] = padded[window:] - padded[:-window]
    return total


def add_lag_features(columns: dict[str, npt.NDArray[np.float64]], config: Config) -> None:
    for window in config.meteo.precip_windows:
        columns[f"precip_cum{window}"] = trailing_sum(columns["precip_sum"], window)

    for source, window in (
        ("temp_mean", config.meteo.temp_window_days),
        ("rh_mean", config.meteo.rh_window_days),
    ):
        columns[f"{source}_{window}d"] = trailing_sum(columns[source], window) / window


def build_backbone(
    config: Config,
) -> tuple[npt.NDArray[Any], dict[str, npt.NDArray[np.float64]], Lattice]:
    """Daily arrays for the whole fetched span, spin-up years included."""
    years = list(fetch_years(config))
    missing = missing_years(config, years)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} year(s) not downloaded: {missing[0]} to {missing[-1]}. "
            "Run `tfire fetch-era5` first."
        )

    lattice: Lattice | None = None
    per_year: list[tuple[npt.NDArray[Any], dict[str, npt.NDArray[np.float64]]]] = []
    for year in years:
        dataset = hourly_span(config, year)
        year_lattice = lattice_of(dataset)
        if lattice is None:
            lattice = year_lattice
            logger.info(
                "ERA5-Land lattice: %d x %d = %d cells",
                lattice.n_rows,
                lattice.n_cols,
                lattice.n_cells,
            )
        elif year_lattice != lattice:
            raise ValueError(f"{year} came back on a different lattice than {years[0]}")

        dates, columns = aggregate_daily(era5_hourly(dataset), config)
        keep = dates.astype("datetime64[Y]").astype(int) + 1970 == year
        per_year.append((dates[keep], {name: values[keep] for name, values in columns.items()}))
        logger.info("%d: %d day(s) aggregated", year, int(keep.sum()))

    if lattice is None:
        raise ValueError("No years to aggregate")

    dates = np.concatenate([part[0] for part in per_year])
    columns = {
        name: np.concatenate([part[1][name] for part in per_year]) for name in per_year[0][1]
    }

    # the opening day's 24 h noon window starts before the record
    incomplete = int(np.isnan(columns["precip_noon24"]).all(axis=1).sum())
    if incomplete != 1:
        logger.error("Expected exactly one day without a noon rain window, found %d", incomplete)
    logger.info("Dropping %s, the first day without a full noon rain window", dates[0])

    return dates[1:], {name: values[1:] for name, values in columns.items()}, lattice


def to_frame(
    dates: npt.NDArray[Any], columns: dict[str, npt.NDArray[np.float64]], lattice: Lattice
) -> pd.DataFrame:
    """Flatten the daily arrays into one row per backbone cell and day."""
    n_days, n_cells = next(iter(columns.values())).shape
    if n_cells != lattice.n_cells:
        raise ValueError(f"{n_cells} columns do not match the {lattice.n_cells}-cell lattice")

    frame = pd.DataFrame(
        {
            "era5_id": np.tile(np.arange(n_cells, dtype="int32"), n_days),
            "date": np.repeat(dates, n_cells).astype("datetime64[us]"),
        }
    )
    for name, values in columns.items():
        stored = values.reshape(-1).astype("float32")
        # a bearing just under 360 rounds up to exactly 360 in float32, which is 0
        frame[name] = wrap_degrees(stored).astype("float32") if name in DIRECTIONS else stored
    return frame


def bilinear_weights(
    lattice: Lattice, longitudes: npt.NDArray[np.float64], latitudes: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """The four surrounding backbone cells and their weights, per point.

    Points outside the lattice clamp onto the edge, which collapses the four weights onto
    the nearest cell rather than extrapolating.
    """
    if lattice.n_rows < 2 or lattice.n_cols < 2:
        raise ValueError(f"Bilinear needs a 2x2 lattice, got {lattice.n_rows}x{lattice.n_cols}")

    step_lon = lattice.longitudes[1] - lattice.longitudes[0]
    step_lat = lattice.latitudes[0] - lattice.latitudes[1]

    col = (longitudes - lattice.longitudes[0]) / step_lon
    row = (lattice.latitudes[0] - latitudes) / step_lat

    left = np.clip(np.floor(col), 0, lattice.n_cols - 2).astype(np.int64)
    top = np.clip(np.floor(row), 0, lattice.n_rows - 2).astype(np.int64)
    fx = np.clip(col - left, 0.0, 1.0)
    fy = np.clip(row - top, 0.0, 1.0)

    corner = top * lattice.n_cols + left
    ids = np.stack(
        [corner, corner + 1, corner + lattice.n_cols, corner + lattice.n_cols + 1], axis=1
    )
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    return ids, weights


def build_era5_weights(config: Config, lattice: Lattice) -> pd.DataFrame:
    """Map every active cell onto the backbone, one row per contributing backbone cell."""
    from pyproj import Transformer

    _, grid = load_grid(config)
    active = grid.loc[grid["is_trentino"]]

    # cell centers are carried into lat/lon only to index the ERA5 lattice; no project
    # data leaves EPSG:25832
    transformer = Transformer.from_crs(config.crs, "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(
        active["x_coordinate"].to_numpy(), active["y_coordinate"].to_numpy()
    )

    ids, weights = bilinear_weights(lattice, longitudes, latitudes)
    frame = pd.DataFrame(
        {
            "cell_id": np.repeat(active["cell_id"].to_numpy(), ids.shape[1]).astype("int32"),
            "era5_id": ids.reshape(-1).astype("int32"),
            "weight": weights.reshape(-1).astype("float64"),
        }
    )
    return frame.loc[frame["weight"] > 0].reset_index(drop=True)


def modeling_window(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Rows the checks apply to: the record and its extension, less the spin-up days ahead of it.

    The spin-up is excluded because its lag columns are legitimately incomplete. The extension
    is not: those days are served, so they are held to the same plausibility bounds.
    """
    dates = frame["date"]
    end = config.meteo.extension_end or config.date_range.end
    return frame.loc[
        (dates >= pd.Timestamp(config.date_range.start)) & (dates <= pd.Timestamp(end))
    ]


def validate_meteo(meteo: pd.DataFrame, lattice: Lattice, config: Config) -> None:
    """Check the acceptance criteria, logging loudly."""
    dates = np.sort(meteo["date"].unique())
    cells = lattice.n_cells
    if len(meteo) != len(dates) * cells:
        logger.error("%d rows, expected %d days x %d cells", len(meteo), len(dates), cells)

    gaps = np.flatnonzero(np.diff(dates) != np.timedelta64(1, "D"))
    if gaps.size:
        logger.error("%d gap(s) in the daily series, first at %s", gaps.size, dates[gaps[0]])
    else:
        logger.info("%s to %s, %d day(s), no gaps", dates[0], dates[-1], len(dates))

    per_cell = meteo.groupby("era5_id")["date"].nunique()
    if per_cell.nunique() != 1:
        logger.error("Backbone cells carry between %d and %d days", per_cell.min(), per_cell.max())

    # the spin-up days exist so the lag features and the FWI codes have history to stand
    # on; their own lag columns are empty by construction and are not a fault
    window = modeling_window(meteo, config)
    logger.info(
        "Checking %d row(s) in %s to %s, %d spin-up row(s) excluded",
        len(window),
        config.date_range.start,
        config.date_range.end,
        len(meteo) - len(window),
    )

    missing = window.isna().sum()
    for name, count in missing[missing > 0].items():
        logger.error("%s: %d missing value(s) inside the modeling window", name, count)

    bounds = {
        "temp_min": (config.meteo.min_temp_c, config.meteo.max_temp_c),
        "temp_max": (config.meteo.min_temp_c, config.meteo.max_temp_c),
        "pres_min": (config.meteo.min_pressure_hpa, config.meteo.max_pressure_hpa),
        "pres_max": (config.meteo.min_pressure_hpa, config.meteo.max_pressure_hpa),
        "rh_min": (0.0, 100.0),
        "rh_max": (0.0, 100.0),
        "precip_sum": (0.0, config.meteo.max_precip_mm),
        "wind_dir_mean": (0.0, 360.0),
        "wind_dir_at_max": (0.0, 360.0),
    }
    for name, (low, high) in bounds.items():
        values = window[name]
        outside = int(((values < low) | (values > high)).sum())
        if outside:
            logger.error(
                "%s: %d value(s) outside [%g, %g], range is [%g, %g]",
                name,
                outside,
                low,
                high,
                values.min(),
                values.max(),
            )


def validate_weights(weights: pd.DataFrame, config: Config) -> None:
    totals = weights.groupby("cell_id")["weight"].sum()
    deviation = float((totals - 1.0).abs().max())
    if deviation > 1e-9:
        logger.error("Interpolation weights deviate from 1.0 by up to %.3g", deviation)

    expected = config.grid.expected_active_cells
    if len(totals) != expected:
        logger.error("%d cell(s) carry weights, expected %d active cells", len(totals), expected)
    else:
        logger.info(
            "%d cell(s) mapped onto %d backbone cell(s), weights sum to 1 within %.3g",
            len(totals),
            weights["era5_id"].nunique(),
            deviation,
        )


def extract_meteo(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `meteo_daily.parquet` and the backbone interpolation weights."""
    out = config.path(config.paths.meteo_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    dates, columns, lattice = build_backbone(config)
    add_lag_features(columns, config)
    meteo = to_frame(dates, columns, lattice)
    validate_meteo(meteo, lattice, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    meteo.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(meteo), meteo.shape[1], out)

    weights_out = config.path(config.paths.era5_weights_out)
    weights = build_era5_weights(config, lattice)
    validate_weights(weights, config)
    weights.to_parquet(weights_out, index=False)
    logger.info("Wrote %d rows to %s", len(weights), weights_out)

    return meteo
