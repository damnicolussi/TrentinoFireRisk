"""Vegetation feature tests. Everything server-side is verified by the pilot."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from tfire.config import Config, load_config
from tfire.features.vegetation import (
    VALUE_NAMES,
    build_climatology,
    forward_fill,
    preceding_composite,
)
from tfire.sources.landsat import (
    _BAND_GROUPS,
    OUTPUT_BANDS,
    band_name,
    cached_months,
    chunk_months,
    chunk_starts,
    parse_band_name,
    reject_mask,
    sensor_spec,
)

MONTHS = [date(2003, month, 1) for month in range(1, 13)]


def series(*values: float) -> np.ndarray:
    """One cell's monthly record, as the (month, cell) shape the fill works on."""
    return np.array(values, dtype="float32").reshape(-1, 1)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("2003-03-17", date(2003, 2, 1)),
        ("2003-03-01", date(2003, 2, 1)),
        ("2003-03-31", date(2003, 2, 1)),
        ("2003-01-01", None),
    ],
    ids=["mid-month", "first-day", "last-day", "before-the-record"],
)
def test_the_composite_covering_the_sample_month_is_never_used(
    sample: str, expected: date | None
) -> None:
    # the failure mode this guards: the composite spanning an ignition also spans the days
    # after it, and the burn scar drops NDVI, so using it leaks the label
    chosen = preceding_composite(pd.Series([pd.Timestamp(sample)]), MONTHS)
    if expected is None:
        assert chosen.isna().all()
    else:
        assert chosen.iloc[0] == pd.Timestamp(expected)


def test_a_band_description_survives_the_round_trip() -> None:
    # a mislabeled month shifts a cell's whole series against the fire dates
    for name in OUTPUT_BANDS:
        for month in (date(1984, 1, 1), date(2003, 10, 1), date(2024, 12, 1)):
            assert parse_band_name(band_name(name, month)) == (name, month)


@pytest.mark.parametrize("description", ["ndvi", "ndvi_2003", "ndvi_20031", "bogus_200310"])
def test_an_unparseable_band_description_raises(description: str) -> None:
    with pytest.raises(ValueError, match="composite band description"):
        parse_band_name(description)


def test_the_qa_mask_covers_the_documented_bits() -> None:
    # a wrong shift silently keeps cloud, which reads as a plausible low NDVI
    assert reject_mask(mask_snow=False) == 0b011111
    assert reject_mask(mask_snow=True) == 0b111111


@pytest.mark.parametrize(
    ("collection", "harmonize"),
    [
        ("LANDSAT/LT04/C02/T1_L2", True),
        ("LANDSAT/LT05/C02/T1_L2", True),
        ("LANDSAT/LE07/C02/T1_L2", True),
        ("LANDSAT/LC08/C02/T1_L2", False),
        ("LANDSAT/LC09/C02/T1_L2", False),
    ],
)
def test_only_the_tm_generation_is_transformed_onto_the_oli_scale(
    collection: str, harmonize: bool
) -> None:
    # Roy's coefficients run ETM+ -> OLI; applying them to OLI too would bias NDVI by a few
    # hundredths in exactly the years that also have the most fires missing
    assert sensor_spec(collection).harmonize is harmonize


def test_an_unknown_collection_raises() -> None:
    with pytest.raises(ValueError, match="Unknown Landsat collection"):
        sensor_spec("LANDSAT/LC10/C02/T1_L2")


def test_the_chunks_tile_the_record_exactly_once(config: Config) -> None:
    months = [month for chunk in chunk_starts(config) for month in chunk_months(config, chunk)]
    first = date(config.date_range.start.year, 1, 1)
    last = date(config.date_range.end.year, 12, 1)
    span = (last.year - first.year) * 12 + 12

    assert len(months) == span
    assert len(set(months)) == span
    assert min(months) == first
    assert max(months) == last


def test_the_band_groups_cover_every_output_band() -> None:
    # the split only runs on the dense years, so a band left out of every group would keep
    # working everywhere else and fail on a year nobody refetches for months
    assert sorted(band for group in _BAND_GROUPS for band in group) == sorted(OUTPUT_BANDS)


def test_months_are_found_across_rasters_of_different_sizes(tmp_path: Path) -> None:
    # 1984-2021 is cached as quarterly rasters and 2022-2024 as monthly ones, because Earth
    # Engine refused the dense years three months at a time. Discovery reads the band
    # descriptions, so it must not assume every file holds the same number of months.
    quarterly = [date(2021, m, 1) for m in (1, 2, 3)]
    monthly = [date(2022, 3, 1)]
    for months in (quarterly, monthly):
        names = [band_name(name, month) for month in months for name in OUTPUT_BANDS]
        path = tmp_path / f"{months[0].year}-{months[0].month:02d}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=len(names),
            dtype="float32",
            crs="EPSG:25832",
            transform=rasterio.Affine(500.0, 0.0, 0.0, 0.0, -500.0, 0.0),
        ) as destination:
            destination.write(np.zeros((len(names), 1, 1), dtype="float32"))
            destination.descriptions = names

    config = load_config()
    config.paths.landsat_raw = tmp_path
    found = cached_months(config)

    assert sorted(found) == [*quarterly, *monthly]
    assert len(set(found.values())) == 2


def test_a_gap_is_carried_forward_until_the_cap_and_then_left_missing() -> None:
    values = {"ndvi": series(0.5, np.nan, np.nan, np.nan, np.nan, np.nan, 0.8)}
    observed = np.array([[True], [False], [False], [False], [False], [False], [True]])

    filled, age = forward_fill(values, observed, MONTHS[:7], max_fill_days=120)

    # January's value survives up to four months out, then the cell goes missing rather than
    # pretending a winter observation describes June
    assert filled["ndvi"][:5, 0] == pytest.approx([0.5, 0.5, 0.5, 0.5, 0.5])
    assert age[:5, 0] == pytest.approx([0, 31, 59, 90, 120])
    assert np.isnan(filled["ndvi"][5, 0])
    assert np.isnan(age[5, 0])
    assert filled["ndvi"][6, 0] == pytest.approx(0.8)
    assert age[6, 0] == pytest.approx(0)


def test_a_cell_with_no_observation_yet_stays_missing() -> None:
    values = {"ndvi": series(np.nan, np.nan, 0.4)}
    observed = np.array([[False], [False], [True]])

    filled, age = forward_fill(values, observed, MONTHS[:3], max_fill_days=120)

    assert np.isnan(filled["ndvi"][:2, 0]).all()
    assert np.isnan(age[:2, 0]).all()
    assert filled["ndvi"][2, 0] == pytest.approx(0.4)


def _monthly_table(years: range, value: float) -> pd.DataFrame:
    rows = [
        {"cell_id": 1, "date": pd.Timestamp(year, 7, 1), "age_days": 0.0}
        | {name: value for name in VALUE_NAMES}
        for year in years
    ]
    return pd.DataFrame(rows)


def test_a_normal_built_from_too_few_years_is_missing_rather_than_confident(
    config: Config,
) -> None:
    minimum = config.vegetation.climatology_min_years
    thin = build_climatology(_monthly_table(range(2000, 2000 + minimum - 1), 0.6), config)
    enough = build_climatology(_monthly_table(range(2000, 2000 + minimum), 0.6), config)

    assert thin["ndvi_mean"].isna().all()
    assert enough["ndvi_mean"].to_numpy() == pytest.approx([0.6])


def test_a_carried_forward_value_never_becomes_part_of_the_normal(config: Config) -> None:
    # one observation repeated across a gap would otherwise count several times and drag the
    # normal toward whatever was last seen
    table = _monthly_table(range(2000, 2010), 0.6)
    stale = table.copy()
    stale["age_days"] = 31.0
    stale[list(VALUE_NAMES)] = 0.1

    climatology = build_climatology(pd.concat([table, stale], ignore_index=True), config)

    assert climatology["ndvi_mean"].to_numpy() == pytest.approx([0.6])
    assert climatology["ndvi_count"].to_numpy() == pytest.approx([10])
