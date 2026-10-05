"""ERA5-Land daily aggregation: de-accumulation, local days, derived-then-aggregated stats."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tfire.config import Config
from tfire.features.meteo import (
    HOURS_PER_DAY,
    add_lag_features,
    aggregate_daily,
    bilinear_weights,
    circular_mean,
    days_since_rain,
    deaccumulate,
    era5_hourly,
    relative_humidity,
    trailing_sum,
    vapor_pressure_deficit,
    wind_speed_direction,
)
from tfire.sources.era5land import Lattice, half_months

_CALM = {
    "t2m": 283.15,
    "d2m": 278.15,
    "sp": 95000.0,
    "tp": 0.0,
    "u10": 0.0,
    "v10": 0.0,
    "swvl1": 0.3,
    "swvl2": 0.35,
}


def hourly(n_hours: int, start: str = "2003-01-01T00", **fields: object) -> xr.Dataset:
    """A single-cell hourly dataset shaped like `era5land.open_year` returns."""
    time = np.arange(
        np.datetime64(start),
        np.datetime64(start) + np.timedelta64(n_hours, "h"),
        np.timedelta64(1, "h"),
    )
    data = {}
    for name, default in _CALM.items():
        values = np.asarray(fields.get(name, default), dtype=float)
        data[name] = (
            ("time", "era5_id"),
            np.broadcast_to(values, (n_hours,)).reshape(-1, 1).copy(),
        )
    return xr.Dataset(data, coords={"time": time, "era5_id": np.array([0])})


@pytest.mark.parametrize(
    ("accumulated", "expected"),
    [
        # hour 0 closes the previous day, hour 1 opens the new one already de-accumulated
        ([6.0, 1.0, 2.0, 3.0], [np.nan, 1.0, 1.0, 1.0]),
        # a dry day: the running total never moves
        ([6.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]),
        # float noise must not surface as negative rainfall
        ([6.0, 1.0, 1.0 - 1e-12, 2.0], [np.nan, 1.0, 0.0, 1.0]),
    ],
    ids=["steady rain", "dry", "noise clamped"],
)
def test_deaccumulate_undoes_the_running_total_since_00_utc(
    accumulated: list[float], expected: list[float]
) -> None:
    utc_hour = np.arange(len(accumulated))
    amounts = deaccumulate(np.array(accumulated).reshape(-1, 1), utc_hour)[:, 0]

    np.testing.assert_allclose(amounts, expected)

    # the failure mode this guards: summing the accumulations instead of the amounts
    assert np.nansum(amounts) < sum(accumulated)


def test_local_day_windowing_follows_the_utc_offset(config: Config) -> None:
    assert config.meteo.utc_offset_hours == 1

    # temperature equals the index, so every aggregate names the hour it came from
    dataset = hourly(72, t2m=273.15 + np.arange(72))
    dates, columns = aggregate_daily(era5_hourly(dataset), config)

    # UTC 23:00 is the first hour of the next local day, so 2003-01-01 is never complete
    assert list(dates.astype(str)) == ["2003-01-02", "2003-01-03"]

    # local noon of 2003-01-02 is the 11:00 UTC record, index 24 + 11
    assert columns["temp_noon"][0, 0] == pytest.approx(35.0)
    assert columns["temp_min"][0, 0] == pytest.approx(23.0)
    assert columns["temp_max"][0, 0] == pytest.approx(46.0)
    assert columns["temp_range"][0, 0] == pytest.approx(23.0)


def test_noon_rain_window_covers_the_24_hours_ending_at_noon(config: Config) -> None:
    # 1 mm in every hour, expressed as ERA5-Land's total since 00 UTC
    accumulated = np.tile(np.arange(HOURS_PER_DAY, dtype=float), 4)
    accumulated[::HOURS_PER_DAY] = HOURS_PER_DAY
    dataset = hourly(96, tp=accumulated / 1000.0)

    _, columns = aggregate_daily(era5_hourly(dataset), config)

    assert np.isnan(columns["precip_noon24"][0, 0])
    assert columns["precip_noon24"][1, 0] == pytest.approx(HOURS_PER_DAY, abs=1e-6)
    assert columns["precip_sum"][1, 0] == pytest.approx(HOURS_PER_DAY, abs=1e-6)


@pytest.mark.parametrize(
    ("temp_c", "dewpoint_c", "expected"),
    [
        (20.0, 20.0, 100.0),
        (20.0, 10.0, 52.54),
        # a dewpoint above the temperature is supersaturation, not humidity above 100
        (5.0, 9.0, 100.0),
    ],
    ids=["saturated", "reference pair", "clipped"],
)
def test_relative_humidity_from_temperature_and_dewpoint(
    temp_c: float, dewpoint_c: float, expected: float
) -> None:
    humidity = relative_humidity(np.array([temp_c]), np.array([dewpoint_c]))
    assert humidity[0] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("u", "v", "speed", "direction"),
    [
        (0.0, -1.0, 1.0, 0.0),
        (-1.0, 0.0, 1.0, 90.0),
        (0.0, 1.0, 1.0, 180.0),
        (1.0, 0.0, 1.0, 270.0),
    ],
    ids=["from north", "from east", "from south", "from west"],
)
def test_wind_direction_is_where_the_wind_blows_from(
    u: float, v: float, speed: float, direction: float
) -> None:
    magnitude, bearing = wind_speed_direction(np.array([u]), np.array([v]))
    assert magnitude[0] == pytest.approx(speed)
    assert bearing[0] == pytest.approx(direction)


def test_circular_mean_wraps_through_north() -> None:
    mean = circular_mean(np.array([[350.0], [10.0]]), axis=0)
    assert mean[0] == pytest.approx(0.0, abs=1e-9)
    assert mean[0] < 360.0


def test_wind_direction_at_max_comes_from_the_windiest_hour(config: Config) -> None:
    # calm westerly all day, one strong northerly gust at 09:00 UTC
    u = np.zeros(48)
    v = np.zeros(48)
    u[:] = 1.0
    u[33], v[33] = 0.0, -20.0
    dataset = hourly(48, u10=u, v10=v)

    _, columns = aggregate_daily(era5_hourly(dataset), config)

    assert columns["wind_speed_max"][0, 0] == pytest.approx(20.0)
    assert columns["wind_dir_at_max"][0, 0] == pytest.approx(0.0)
    assert columns["wind_dir_mean"][0, 0] == pytest.approx(270.0, abs=15.0)


def test_humidity_is_derived_hourly_then_averaged(config: Config) -> None:
    # a wide diurnal swing around a constant dewpoint, where the two orderings diverge
    swing = 15.0 + 15.0 * np.sin(np.arange(48) * 2 * np.pi / HOURS_PER_DAY)
    dataset = hourly(48, t2m=273.15 + swing, d2m=273.15 + 5.0)

    _, columns = aggregate_daily(era5_hourly(dataset), config)

    hours = swing[23:47]
    hourly_first = relative_humidity(hours, np.full(hours.size, 5.0)).mean()
    aggregate_first = relative_humidity(np.array([hours.mean()]), np.array([5.0]))[0]

    assert columns["rh_mean"][0, 0] == pytest.approx(hourly_first)
    assert abs(hourly_first - aggregate_first) > 1.0


def test_wind_speed_is_derived_hourly_then_averaged(config: Config) -> None:
    # the wind reverses at noon, so the daily mean of the components is nearly zero
    u = np.where(np.arange(48) % HOURS_PER_DAY < 12, 5.0, -5.0)
    dataset = hourly(48, u10=u)

    _, columns = aggregate_daily(era5_hourly(dataset), config)

    assert columns["wind_speed_mean"][0, 0] == pytest.approx(5.0)
    assert abs(np.hypot(u[23:47].mean(), 0.0)) < 1.0


@pytest.mark.parametrize(
    ("temp_c", "dewpoint_c", "expected"),
    [
        (20.0, 20.0, 0.0),
        # 23.34 hPa saturated at 20 C against 12.26 hPa actual at a 10 C dewpoint
        (20.0, 10.0, 11.07),
        (5.0, 9.0, 0.0),
    ],
    ids=["saturated", "reference pair", "supersaturated"],
)
def test_vapor_pressure_deficit_from_temperature_and_dewpoint(
    temp_c: float, dewpoint_c: float, expected: float
) -> None:
    deficit = vapor_pressure_deficit(np.array([temp_c]), np.array([dewpoint_c]))
    assert deficit[0] == pytest.approx(expected, abs=0.01)


def test_the_deficit_peak_is_derived_hourly_then_maximized(cell_config: Config) -> None:
    # the same swing as the humidity test: the deficit of the daily mean misses the afternoon
    swing = 15.0 + 15.0 * np.sin(np.arange(48) * 2 * np.pi / HOURS_PER_DAY)
    dataset = hourly(48, t2m=273.15 + swing, d2m=273.15 + 5.0)

    _, columns = aggregate_daily(era5_hourly(dataset), cell_config)

    hours = swing[23:47]
    hourly_peak = vapor_pressure_deficit(hours, np.full(hours.size, 5.0)).max()
    from_daily_max = vapor_pressure_deficit(np.array([hours.max()]), np.array([5.0]))[0]
    from_daily_mean = vapor_pressure_deficit(np.array([hours.mean()]), np.array([5.0]))[0]

    assert columns["vpd_max"][0, 0] == pytest.approx(hourly_peak)
    assert hourly_peak == pytest.approx(from_daily_max)
    assert hourly_peak - from_daily_mean > 10.0


def test_days_since_rain_counts_from_the_last_wet_day_and_stops_at_the_cap() -> None:
    precip = np.zeros((10, 3))
    precip[2, 0] = 5.0
    precip[5, 0] = 0.4  # under the threshold, still a dry day
    precip[0, 1] = 1.0

    counts = days_since_rain(precip, threshold_mm=1.0, cap=4)

    # unknown until it rains or has been dry for the whole cap
    np.testing.assert_array_equal(counts[:2, 0], [np.nan, np.nan])
    np.testing.assert_array_equal(counts[2:, 0], [0, 1, 2, 3, 4, 4, 4, 4])
    np.testing.assert_array_equal(counts[:, 1], [0, 1, 2, 3, 4, 4, 4, 4, 4, 4])
    np.testing.assert_array_equal(counts[:, 2], [np.nan] * 3 + [4] * 7)


def test_a_short_history_gives_the_same_count_once_it_covers_the_cap() -> None:
    """The served path starts from a spin-up, the cache from the whole record."""
    rng = np.random.default_rng(0)
    precip = np.where(rng.uniform(size=(400, 20)) < 0.08, 3.0, 0.0)
    cap = 60

    whole = days_since_rain(precip, 1.0, cap)
    for start in (0, 37, 250):
        served = days_since_rain(precip[start:], 1.0, cap)
        np.testing.assert_array_equal(served[cap - 1 :], whole[start + cap - 1 :])


def test_the_dry_spell_cap_fits_inside_the_served_spin_up(config: Config) -> None:
    assert config.meteo.dry_spell_cap_days <= config.forecast.spinup_days


def test_with_cell_scale_off_the_backbone_carries_only_what_v2_was_fitted_on(
    config: Config, cell_config: Config
) -> None:
    dataset = hourly(72)
    _, off = aggregate_daily(era5_hourly(dataset), config)
    add_lag_features(off, config)
    _, on = aggregate_daily(era5_hourly(dataset), cell_config)
    add_lag_features(on, cell_config)

    added = set(on) - set(off)
    assert {"vpd_max", "days_since_rain"} <= added
    assert not {"vpd_max", "days_since_rain"} & set(off)
    assert all(name in {"vpd_max", "days_since_rain"} or "_dt" in name for name in added)


@pytest.mark.parametrize("window", [7, 15, 30], ids=["7d", "15d", "30d"])
def test_trailing_sum_ends_on_the_current_day(window: int) -> None:
    values = np.arange(40, dtype=float).reshape(-1, 1)
    total = trailing_sum(values, window)

    assert np.isnan(total[: window - 1, 0]).all()
    assert total[window - 1, 0] == pytest.approx(values[:window, 0].sum())
    assert total[-1, 0] == pytest.approx(values[-window:, 0].sum())

    # a later value must not change an earlier window
    lifted = values.copy()
    lifted[-1] = 1e6
    assert trailing_sum(lifted, window)[-2, 0] == pytest.approx(total[-2, 0])


@pytest.mark.parametrize(
    ("half", "expected"), [(1, [1, 2, 3, 4, 5, 6]), (2, [7, 8, 9, 10, 11, 12])]
)
def test_half_months_partition_the_year(half: int, expected: list[int]) -> None:
    assert half_months(half) == expected


LATTICE = Lattice(np.array([46.6, 46.5, 46.4]), np.array([10.4, 10.5, 10.6]))


@pytest.mark.parametrize(
    ("longitude", "latitude", "expected"),
    [
        # exactly on the middle node
        (10.5, 46.5, {4: 1.0}),
        # the center of the north-west square, split four ways
        (10.45, 46.55, {0: 0.25, 1: 0.25, 3: 0.25, 4: 0.25}),
        # halfway along the top edge
        (10.45, 46.6, {0: 0.5, 1: 0.5}),
        # outside the lattice, clamped onto the nearest corner
        (10.0, 47.0, {0: 1.0}),
    ],
    ids=["on a node", "square center", "on an edge", "outside"],
)
def test_bilinear_weights_place_a_point_on_the_backbone(
    longitude: float, latitude: float, expected: dict[int, float]
) -> None:
    ids, weights = bilinear_weights(LATTICE, np.array([longitude]), np.array([latitude]))

    assert weights.sum() == pytest.approx(1.0)
    placed = {
        int(cell): float(weight)
        for cell, weight in zip(ids[0], weights[0], strict=True)
        if weight > 0
    }
    assert placed == pytest.approx(expected)
