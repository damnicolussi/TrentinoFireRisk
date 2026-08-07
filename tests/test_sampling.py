"""Label levels, the exclusion set and the negative draw."""

from __future__ import annotations

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tfire.config import Config, DateRangeConfig
from tfire.grid import GridSpec
from tfire.sampling import (
    LEVEL_BUFFER,
    LEVEL_POLYGON,
    build_positive_rows,
    event_dates,
    exclusion_pairs,
    sample_negatives,
    unique_exclusions,
)

CRS = "EPSG:25832"
SMALL = GridSpec(xmin=0.0, ymax=5000.0, n_cols=5, n_rows=5, resolution_m=1000, crs=CRS)

# well inside cell 12, the middle of the 5x5 grid
BURN = box(2100.0, 2100.0, 2900.0, 2900.0)
FIRST_DAY = pd.Timestamp("2020-06-10")
LAST_DAY = pd.Timestamp("2020-06-12")


def tuned(config: Config, **sampling: object) -> Config:
    """The real config with the resolution, the window and the sampling knobs swapped out."""
    return config.model_copy(
        update={
            "resolution_m": SMALL.resolution_m,
            "date_range": DateRangeConfig(start=date(2020, 6, 1), end=date(2020, 6, 30)),
            "sampling": config.sampling.model_copy(update=sampling),
        }
    )


def synthetic_grid(active: set[int], non_burnable: set[int]) -> pd.DataFrame:
    cell_id = np.arange(SMALL.n_cells)
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "is_trentino": np.isin(cell_id, sorted(active)),
            "is_non_burnable": np.isin(cell_id, sorted(non_burnable)),
        }
    )


@pytest.mark.parametrize(
    ("buffer_cells", "ring"),
    [(0, [12]), (1, [6, 7, 8, 11, 12, 13, 16, 17, 18])],
    ids=["polygon_only", "one_cell_ring"],
)
def test_the_two_exclusion_levels_cover_the_polygon_and_its_ring(
    config: Config, buffer_cells: int, ring: list[int]
) -> None:
    cfg = tuned(config, buffer_cells=buffer_cells)
    fires = pd.DataFrame({"fire_id": [1], "start_date": [FIRST_DAY], "end_date": [LAST_DAY]})
    polygons = gpd.GeoDataFrame({"fire_id": [1]}, geometry=[BURN], crs=CRS)

    pairs = exclusion_pairs(fires, polygons, SMALL, cfg)
    burned = pairs.loc[pairs["level"] == LEVEL_POLYGON]
    buffered = pairs.loc[pairs["level"] == LEVEL_BUFFER]

    slack = pd.Timedelta(days=cfg.sampling.buffer_days)
    assert sorted(burned["cell_id"].unique()) == [12]
    assert sorted(buffered["cell_id"].unique()) == ring
    assert pd.DatetimeIndex(burned["date"].unique()).equals(pd.date_range(FIRST_DAY, LAST_DAY))
    assert pd.DatetimeIndex(buffered["date"].unique()).equals(
        pd.date_range(FIRST_DAY - slack, LAST_DAY + slack)
    )

    exclusions = unique_exclusions(pairs)
    assert len(exclusions) == len(ring) * len(pd.date_range(FIRST_DAY - slack, LAST_DAY + slack))
    assert int((exclusions["level"] == LEVEL_POLYGON).sum()) == 3


def test_no_negative_lands_on_an_excluded_or_unusable_cell(config: Config) -> None:
    cfg = tuned(config)
    grid = synthetic_grid(active=set(range(10)), non_burnable={3, 4})
    days = pd.date_range(cfg.date_range.start, cfg.date_range.end)

    # cells 0, 1, 2 and 5 are excluded outright, leaving 6, 7, 8 and 9 usable
    blocked = [0, 1, 2, 5]
    exclusions = pd.DataFrame(
        {
            "cell_id": np.repeat(blocked, len(days)),
            "date": np.tile(days.to_numpy(), len(blocked)),
            "level": LEVEL_POLYGON,
        }
    )

    negatives = sample_negatives(grid, exclusions, 100, cfg, np.random.default_rng(0))

    assert len(negatives) == 100
    assert set(negatives["cell_id"]) == {6, 7, 8, 9}
    assert not negatives.duplicated().any()
    assert not (
        pd.MultiIndex.from_frame(negatives[["cell_id", "date"]])
        .isin(pd.MultiIndex.from_frame(exclusions[["cell_id", "date"]]))
        .any()
    )

    with pytest.raises(ValueError, match="available"):
        sample_negatives(grid, exclusions, len(days) * 4 + 1, cfg, np.random.default_rng(0))


def test_the_seed_alone_decides_which_cell_days_are_drawn(config: Config) -> None:
    cfg = tuned(config)
    grid = synthetic_grid(active=set(range(SMALL.n_cells)), non_burnable=set())
    exclusions = pd.DataFrame({"cell_id": [0], "date": [FIRST_DAY], "level": [LEVEL_POLYGON]})

    def draw(seed: int) -> pd.DataFrame:
        return sample_negatives(grid, exclusions, 50, cfg, np.random.default_rng(seed))

    assert draw(cfg.project.random_seed).equals(draw(cfg.project.random_seed))
    assert not draw(cfg.project.random_seed).equals(draw(cfg.project.random_seed + 1))


def test_fires_on_one_cell_day_collapse_onto_the_largest(config: Config) -> None:
    cfg = tuned(config)
    fires = pd.DataFrame(
        {
            "fire_id": [1, 2, 3, 4],
            "cell_id": [12, 12, 12, 0],
            "start_date": [FIRST_DAY, FIRST_DAY, LAST_DAY, FIRST_DAY],
            "area_ha": [0.5, 7.0, 2.0, 3.0],
        }
    )

    positives = build_positive_rows(fires, synthetic_grid(active={12}, non_burnable=set()), cfg)

    assert len(positives) == 2
    assert 4 not in set(positives["fire_id"])  # cell 0 is outside the boundary

    shared = positives.loc[positives["date"] == FIRST_DAY].iloc[0]
    assert shared["fire_id"] == 2
    assert shared["n_fires"] == 2


@pytest.mark.parametrize(
    ("end_datetime", "last_day"),
    [
        (pd.Timestamp("2020-06-12 08:00"), LAST_DAY),
        (pd.NaT, FIRST_DAY),
        (pd.Timestamp("2020-06-09 12:00"), FIRST_DAY),
    ],
    ids=["normalized", "missing_end", "end_before_start"],
)
def test_the_event_span_never_inverts(end_datetime: pd.Timestamp, last_day: pd.Timestamp) -> None:
    fires = pd.DataFrame({"ignition_date": [FIRST_DAY], "end_datetime": [end_datetime]})

    start, end = event_dates(fires)

    assert start.iloc[0] == FIRST_DAY
    assert end.iloc[0] == last_day
