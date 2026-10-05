"""The lapse correction: offsets, the rung interpolation, and what it costs against exact hours."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.features.lapse import (
    OFFSET_COLUMN,
    apply_lapse,
    blend,
    cell_offsets,
    lapse_columns,
    rung_columns,
)
from tfire.features.meteo import (
    HOURS_PER_DAY,
    NOON_HOUR_LST,
    add_lag_features,
    aggregate_daily,
    era5_hourly,
    relative_humidity,
    vapor_pressure_deficit,
)

from .test_meteo import hourly


def test_a_cell_above_the_backbone_terrain_is_colder() -> None:
    offsets = cell_offsets(np.array([1500.0, 500.0]), np.array([500.0, 1500.0]), 6.5)
    np.testing.assert_allclose(offsets, [-6.5, 6.5])


def test_the_blend_is_exact_on_a_rung_and_linear_between() -> None:
    rungs = np.array([-2.0, 0.0, 2.0])
    values = np.array([[10.0, 20.0, 40.0]] * 4)
    blended = blend(values, rungs, np.array([-2.0, 0.0, 1.0, 2.0]))
    np.testing.assert_allclose(blended, [10.0, 20.0, 30.0, 40.0])

    with pytest.raises(ValueError, match="outside the rungs"):
        blend(values[:1], rungs, np.array([2.5]))


def _days(config: Config) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Ten days of a wide diurnal swing under a slowly drifting dewpoint, one backbone cell.

    The hourly arrays come back cut to the local days `aggregate_daily` kept, one row per day.
    """
    hours = np.arange(10 * HOURS_PER_DAY + 23)
    temp = 12.0 + 11.0 * np.sin(hours * 2 * np.pi / HOURS_PER_DAY) + 0.05 * hours / 24
    dew = 3.0 + 2.0 * np.sin(hours * 2 * np.pi / (5 * HOURS_PER_DAY))
    fields = era5_hourly(hourly(hours.size, t2m=273.15 + temp, d2m=273.15 + dew))
    _, columns = aggregate_daily(fields, config)
    add_lag_features(columns, config)

    local = (fields.times + np.timedelta64(config.meteo.utc_offset_hours, "h")).astype("int64")
    start = int(np.flatnonzero(local % HOURS_PER_DAY == 0)[0])
    n_days = columns["temp_mean"].shape[0]
    stop = start + n_days * HOURS_PER_DAY
    shape = (n_days, HOURS_PER_DAY)
    return (
        columns,
        fields.temp_c[start:stop, 0].reshape(shape),
        fields.dewpoint_c[start:stop, 0].reshape(shape),
    )


def _frame(columns: dict[str, np.ndarray], offset: float) -> pd.DataFrame:
    frame = pd.DataFrame({name: values[:, 0].astype("float32") for name, values in columns.items()})
    frame[OFFSET_COLUMN] = offset
    return frame


def test_at_a_zero_offset_the_lapse_columns_are_the_backbone_ones(cell_config: Config) -> None:
    columns, _, _ = _days(cell_config)
    out = apply_lapse(_frame(columns, 0.0), cell_config)

    for name in lapse_columns(cell_config):
        base = name.removesuffix("_lapse")
        np.testing.assert_allclose(out[name], out[base], rtol=0, atol=1e-4, err_msg=name)
    assert not set(rung_columns(cell_config)) & set(out.columns)
    assert OFFSET_COLUMN not in out.columns


@pytest.mark.parametrize("offset", np.linspace(-9.9, 7.9, 37).round(2).tolist())
def test_the_rungs_stay_within_a_fraction_of_a_point_of_exact_hours(
    cell_config: Config, offset: float
) -> None:
    columns, temp, dew = _days(cell_config)
    out = apply_lapse(_frame(columns, offset), cell_config)

    humidity = relative_humidity(temp + offset, dew)
    deficit = vapor_pressure_deficit(temp + offset, dew)

    np.testing.assert_allclose(out["temp_mean_lapse"], temp.mean(axis=1) + offset, atol=1e-4)
    for name, exact, tolerance in (
        ("rh_mean_lapse", humidity.mean(axis=1), 0.3),
        ("rh_min_lapse", humidity.min(axis=1), 0.3),
        ("rh_max_lapse", humidity.max(axis=1), 0.3),
        ("rh_noon_lapse", humidity[:, NOON_HOUR_LST], 0.3),
        ("vpd_max_lapse", deficit.max(axis=1), 0.1),
    ):
        np.testing.assert_allclose(out[name], exact, rtol=0, atol=tolerance, err_msg=name)
