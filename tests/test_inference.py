"""Operational inference: provider routing, column alignment, and the raster round trip."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.features.meteo import HourlyFields, aggregate_daily, era5_hourly
from tfire.grid import GridSpec
from tfire.inference import FeatureUnavailableError, plan_days
from tfire.models.trentino import align_columns
from tfire.raster import read_cell_bands, write_cell_bands
from tfire.sources.forecast import ForecastError, plan_span

from .test_meteo import hourly

TODAY = date(2026, 8, 16)


def test_the_forecast_adapter_aggregates_exactly_like_the_era5_one(config: Config) -> None:
    """A unit slip in the adapter yields a plausible map, not an error, so it is pinned here."""
    dataset = hourly(
        96,
        t2m=273.15 + 10 + np.arange(96) % 12,
        d2m=273.15 + 5.0,
        sp=95000.0,
        tp=np.tile(np.arange(24, dtype=float), 4) / 1000.0,
        u10=3.0,
        v10=-4.0,
    )
    era5 = era5_hourly(dataset)

    # the same hours as a provider that already ships physical units would hand them over
    equivalent = HourlyFields(
        times=era5.times,
        temp_c=era5.temp_c,
        dewpoint_c=era5.dewpoint_c,
        pres_hpa=era5.pres_hpa,
        precip_mm=era5.precip_mm,
        wind_speed=era5.wind_speed,
        wind_dir=era5.wind_dir,
    )

    reference, expected = aggregate_daily(era5, config)
    dates, actual = aggregate_daily(equivalent, config)

    assert list(dates) == list(reference)
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, err_msg=name)

    # the ERA5 conversions themselves. u=3 v=-4 blows toward the south-east at 5 m/s, and a
    # bearing names where wind comes from, so it reads as north-west
    assert expected["pres_mean"][0, 0] == pytest.approx(950.0)
    assert expected["wind_speed_mean"][0, 0] == pytest.approx(5.0)
    assert expected["wind_dir_mean"][0, 0] == pytest.approx(323.13, abs=0.01)


def test_a_single_day_aligns_onto_the_stored_columns() -> None:
    """One date carries one season, so three of the four indicators are absent by construction."""
    stored = ["fwi", "season_autumn", "season_spring", "season_summer", "season_winter"]
    features = pd.DataFrame({"season_summer": [1.0, 1.0], "fwi": [12.0, 3.0]})

    aligned = align_columns(features, stored)

    assert list(aligned.columns) == stored
    assert aligned["season_summer"].tolist() == [1.0, 1.0]
    assert aligned["season_winter"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"season_summer": [1.0]}), "missing"),
        (pd.DataFrame({"fwi": [1.0], "season_summer": [1.0], "stowaway": [1.0]}), "undeclared"),
    ],
    ids=["a real feature is absent", "an undeclared column rode along"],
)
def test_alignment_refuses_a_matrix_that_is_not_the_stored_one(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        align_columns(frame, ["fwi", "season_summer", "season_winter"])


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2000, 6, 1), ["cached"]),
        (date(1984, 1, 1), ["cached"]),
        (date(2024, 12, 31), ["cached"]),
        (date(2025, 1, 1), ["archive"]),
        (TODAY, ["archive", "forecast"]),
        (TODAY + timedelta(days=10), ["archive", "forecast"]),
    ],
    ids=["mid record", "first day", "last cached day", "just past it", "today", "the horizon"],
)
def test_every_date_in_range_finds_a_provider(
    config: Config, day: date, expected: list[str]
) -> None:
    plan = plan_days(config, [day], today=TODAY)

    if expected == ["cached"]:
        assert plan.meteo_from_cache
    else:
        assert [piece.provider for piece in plan.segments] == expected


@pytest.mark.parametrize(
    "day",
    [TODAY + timedelta(days=11), date(1983, 12, 31)],
    ids=["past the forecast horizon", "before the record"],
)
def test_a_date_no_source_covers_names_what_it_cannot_build(config: Config, day: date) -> None:
    with pytest.raises(FeatureUnavailableError) as caught:
        plan_days(config, [day], today=TODAY)

    assert "fwi" in caught.value.features
    assert str(day) in str(caught.value)


def test_the_spin_up_is_taken_from_the_archive_not_the_forecast(config: Config) -> None:
    """The forecast endpoint's own past is shorter than the spin-up, so the seam has to fall
    inside the archive's reach or the codes would start from nothing."""
    segments = plan_span(
        config, TODAY - timedelta(days=config.forecast.spinup_days), TODAY, TODAY
    )

    assert [piece.provider for piece in segments] == ["archive", "forecast"]
    assert segments[0].end + timedelta(days=1) == segments[1].start


def test_a_span_past_the_horizon_is_refused_before_anything_is_fetched(config: Config) -> None:
    with pytest.raises(ForecastError, match="horizon"):
        plan_span(config, TODAY, TODAY + timedelta(days=30), TODAY)


def test_a_written_raster_reads_back_cell_for_cell(tmp_path: Path) -> None:
    """A transposed reshape would mirror the province and still produce a plausible map."""
    spec = GridSpec(
        xmin=612000.0, ymax=5157500.0, n_cols=4, n_rows=3, resolution_m=500, crs="EPSG:25832"
    )
    values = np.arange(spec.n_cells, dtype="float64")
    bands = {"probability": values, "rank": values[::-1].copy()}

    path = write_cell_bands(spec, tmp_path / "risk.tif", bands)
    back = read_cell_bands(spec, path)

    for name, expected in bands.items():
        np.testing.assert_allclose(back[name], expected, err_msg=name)
