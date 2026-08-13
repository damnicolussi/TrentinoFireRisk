"""Sub-point distance statistics, POI capacity aggregation and calendar features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from tfire.config import Config
from tfire.features.human import calendar_features, poi_frame, sub_point_distance_stats
from tfire.grid import GridSpec
from tfire.sources import osm

CRS = "EPSG:25832"

# 2 cells of 10 m, cell_id 0 west of cell_id 1
LINE_GRID = GridSpec(xmin=0.0, ymax=10.0, n_cols=2, n_rows=1, resolution_m=10, crs=CRS)

# 4 cells of 10 m in a row, used for the POI aggregation test
POI_GRID = GridSpec(xmin=0.0, ymax=10.0, n_cols=4, n_rows=1, resolution_m=10, crs=CRS)


def test_sub_point_distance_stats_matches_a_hand_computed_grid() -> None:
    """a vertical line at x=0: distance from a sub-point is exactly its x coordinate. With
    n_per_side=2, sub-point offsets are 2.5 and 7.5 within each 10 m cell.
    """
    line = gpd.GeoSeries([LineString([(0.0, -100.0), (0.0, 100.0)])], crs=CRS)

    mean, std = sub_point_distance_stats(LINE_GRID, np.array([0, 1]), line, n_per_side=2)

    np.testing.assert_allclose(mean, [5.0, 15.0])
    np.testing.assert_allclose(std, [2.5, 2.5])


def test_sub_point_distance_stats_is_nan_when_the_layer_is_empty() -> None:
    empty = gpd.GeoSeries([], crs=CRS)

    mean, std = sub_point_distance_stats(LINE_GRID, np.array([0, 1]), empty, n_per_side=2)

    assert np.isnan(mean).all()
    assert np.isnan(std).all()


def test_poi_frame_counts_both_kinds_of_presence_over_the_same_cell_area(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cell 0: a hotel and an alpine_hut. The hut is lodging and a day destination at once, so
        it counts toward both features: 2 accommodation and 1 recreational over the cell area.
    cell 1: a parking POI (recreational only) and a bed_and_breakfast (accommodation only).
    cell 2: a hostel carrying no size tag, which still has to register as lodging.
    cell 3: no POI at all -> zero on both, never null.
    """
    poi = gpd.GeoDataFrame(
        {
            "tourism": ["hotel", "alpine_hut", np.nan, "bed_and_breakfast", "hostel"],
            "amenity": [np.nan, np.nan, "parking", np.nan, np.nan],
        },
        geometry=[Point(5, 5), Point(6, 6), Point(15, 5), Point(12, 8), Point(25, 5)],
        crs=CRS,
    )
    monkeypatch.setattr(osm, "fetch_poi", lambda config: poi)

    frame = poi_frame(POI_GRID, np.array([0, 1, 2, 3]), config).set_index("cell_id")

    per_km2 = 1 / (POI_GRID.resolution_m / 1000) ** 2
    np.testing.assert_allclose(frame["accommodation_density"], np.array([2, 1, 1, 0]) * per_km2)
    np.testing.assert_allclose(frame["recreational_poi_density"], np.array([1, 1, 0, 0]) * per_km2)
    assert not frame.isna().to_numpy().any()


@pytest.mark.parametrize(
    ("date", "month", "season", "day_of_week", "is_weekend", "is_holiday"),
    [
        ("2024-08-15", 8, "summer", 3, False, True),  # Ferragosto, a Thursday
        ("2024-08-17", 8, "summer", 5, True, False),  # a plain Saturday
        ("2023-12-31", 12, "winter", 6, True, False),  # December wraps into winter
    ],
    ids=["holiday_weekday", "plain_weekend", "december_is_winter"],
)
def test_calendar_features_on_known_dates(
    date: str, month: int, season: str, day_of_week: int, is_weekend: bool, is_holiday: bool
) -> None:
    frame = calendar_features(pd.DatetimeIndex([date]))

    row = frame.iloc[0]
    assert row["month"] == month
    assert row["season"] == season
    assert row["day_of_week"] == day_of_week
    assert row["is_weekend"] == is_weekend
    assert row["is_holiday"] == is_holiday
