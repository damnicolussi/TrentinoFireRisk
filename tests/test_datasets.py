"""Backbone interpolation, the nearest-cell rule, and registry enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.datasets import decade_labels, interpolate_meteo, nearest_backbone
from tfire.features.landcover import nearest_edition
from tfire.features.registry import FeatureSpec, Registry, validate_frame

DAY = pd.Timestamp("2003-07-15")

SAMPLES = pd.DataFrame({"sample_id": [0], "cell_id": [0], "date": [DAY]})

WEIGHTS = pd.DataFrame(
    {
        "cell_id": [0, 0, 0, 0],
        "era5_id": [10, 11, 12, 13],
        "weight": [0.4, 0.3, 0.2, 0.1],
    }
)


def backbone(*values: float) -> pd.DataFrame:
    return pd.DataFrame({"era5_id": [10, 11, 12, 13], "date": DAY, "temp_mean": list(values)})


def test_meteo_is_the_weighted_mean_of_the_four_surrounding_backbone_cells() -> None:
    """A merge that drops or misaligns a neighbor still returns a temperature that looks like one"""
    result = interpolate_meteo(SAMPLES, WEIGHTS, backbone(10.0, 20.0, 30.0, 40.0))

    assert result.loc[0, "temp_mean"] == pytest.approx(20.0)


def test_a_null_neighbor_makes_the_sample_null_rather_than_a_partial_sum() -> None:
    """Summing over three of four weights yields a plausible number that is 10% too low."""
    result = interpolate_meteo(SAMPLES, WEIGHTS, backbone(10.0, 20.0, 30.0, np.nan))

    assert result["temp_mean"].isna().iloc[0]


def test_interpolation_refuses_weights_that_do_not_sum_to_one() -> None:
    partial = WEIGHTS.drop(index=3)

    with pytest.raises(ValueError, match="sum to 1"):
        interpolate_meteo(SAMPLES, partial, backbone(10.0, 20.0, 30.0, 40.0))


def test_the_nearest_backbone_cell_is_the_heaviest_one_and_ties_go_to_the_lowest_id() -> None:
    weights = pd.DataFrame(
        {
            "cell_id": [0, 0, 1, 1],
            "era5_id": [5, 3, 7, 2],
            "weight": [0.5, 0.5, 0.6, 0.4],
        }
    )

    chosen = nearest_backbone(weights).set_index("cell_id")["era5_id"]

    assert chosen.to_dict() == {0: 3, 1: 7}


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2002-06-01", 2000),
        ("2003-01-01", 2005),
        ("2007-12-31", 2005),
        ("2008-01-01", 2010),
        ("1984-01-01", 2000),
        ("2024-12-31", 2020),
    ],
    ids=["before_a_tie", "tie_2002", "before_the_next_tie", "tie_2007", "before", "after"],
)
def test_a_sample_joins_the_nearest_worldpop_year(date: str, expected: int) -> None:
    """WorldPop reuses the CORINE rule, so the ties have to fall the same way: earlier wins."""
    years = (2000, 2005, 2010, 2015, 2020)

    assert int(nearest_edition(pd.Series(pd.to_datetime([date])), years).iloc[0]) == expected


def test_decades_are_labeled_by_the_years_they_hold() -> None:
    years = pd.Series([1984, 1989, 1990, 2020, 2024])

    assert list(decade_labels(years)) == [
        "1984-1989",
        "1984-1989",
        "1990-1990",
        "2020-2024",
        "2020-2024",
    ]


def registry() -> Registry:
    return Registry(
        [
            FeatureSpec(
                name="ndvi",
                category="vegetation",
                source="test",
                dtype="float32",
                temporal="dynamic",
                min=-1.0,
                max=1.0,
            )
        ]
    )


def test_an_undeclared_column_is_rejected(config: Config) -> None:
    frame = pd.DataFrame({"ndvi": pd.Series([0.5], dtype="float32"), "surprise": [1.0]})

    with pytest.raises(ValueError, match="registry violation"):
        validate_frame(frame, registry(), config)


def test_a_value_outside_its_declared_range_is_rejected(config: Config) -> None:
    frame = pd.DataFrame({"ndvi": pd.Series([1.4], dtype="float32")})

    with pytest.raises(ValueError, match="registry violation"):
        validate_frame(frame, registry(), config)


def test_a_config_reference_resolves_to_the_master_config(config: Config) -> None:
    """The bound lives in config.yaml; copying it into the registry is how the two drift."""
    spec = FeatureSpec(
        name="elevation_mean",
        category="topography",
        source="test",
        dtype="float32",
        temporal="static",
        max="config:topography.max_elevation_m",
    )

    assert spec.bounds(config) == (None, config.topography.max_elevation_m)
