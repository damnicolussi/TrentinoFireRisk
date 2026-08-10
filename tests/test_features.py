"""Static feature extraction: CLC proportions, edition choice, aspect sectors, water."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from tfire.config import Config
from tfire.features.geography import waterbody_fraction
from tfire.features.landcover import (
    CLC_LEVEL_3,
    CLC_NODATA_CODE,
    class_proportions,
    group_by_level,
    nearest_edition,
    proportion_frame,
)
from tfire.grid import GridSpec
from tfire.sources.dem import aspect_sectors

CRS = "EPSG:25832"
EDITIONS = (1990, 2000, 2006, 2012, 2018)

# 4x4 pixels under a 2x2 grid of cells, the south-east cell entirely NODATA
PIXELS = np.array(
    [
        [24, 24, 23, 41],
        [24, 12, 23, 41],
        [1, 1, -128, -128],
        [2, 18, -128, -128],
    ],
    dtype="int16",
)
BLOCKS = PIXELS.reshape(2, 2, 2, 2)
NODATA = -128


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("1984-06-01", 1990),
        ("1995-12-31", 1990),
        ("1996-01-01", 2000),
        ("2003-08-15", 2000),
        ("2004-08-15", 2006),
        ("2009-12-31", 2006),
        ("2010-01-01", 2012),
        ("2015-07-01", 2012),
        ("2016-07-01", 2018),
        ("2024-12-31", 2018),
    ],
    ids=[
        "before_the_first_edition",
        "tie_1995",
        "after_the_1995_tie",
        "tie_2003",
        "after_the_2003_tie",
        "tie_2009",
        "after_the_2009_tie",
        "tie_2015",
        "after_the_2015_tie",
        "last_day_in_range",
    ],
)
def test_a_sample_joins_the_nearest_edition_in_time(date: str, expected: int) -> None:
    """The four tie years are the boundaries, and each has to fall the earlier way."""
    dates = pd.Series(pd.to_datetime([date]))

    assert int(nearest_edition(dates, EDITIONS).iloc[0]) == expected


def test_the_code_table_matches_the_legend_shipped_with_the_rasters(config: Config) -> None:
    """A mistyped 3-digit code moves a class into the wrong aggregate, silently."""
    legend = config.path(config.paths.corine_raw) / config.corine.editions[2018]
    legend = legend.with_name(legend.name + ".vat.dbf")
    if not legend.is_file():
        pytest.skip(f"CORINE raw data not present: {legend}")

    table = gpd.read_file(legend)
    shipped = dict(zip(table["Value"].astype(int), table["CODE_18"].astype(int), strict=True))

    assert shipped.pop(CLC_NODATA_CODE) == 999
    assert {code: CLC_LEVEL_3[code] for code in shipped} == shipped


@pytest.mark.parametrize(("level", "n_groups"), [(2, 15), (1, 5)])
def test_every_grid_code_lands_in_exactly_one_aggregate(level: int, n_groups: int) -> None:
    groups = group_by_level(level)
    members = [code for group in groups.values() for code in group]

    assert len(groups) == n_groups
    assert sorted(members) == sorted(CLC_LEVEL_3)


def test_class_proportions_are_shares_of_the_covered_pixels() -> None:
    proportions = class_proportions(BLOCKS, NODATA)

    assert proportions.shape == (4, len(CLC_LEVEL_3))
    assert proportions[0, 24 - 1] == 0.75
    assert proportions[0, 12 - 1] == 0.25
    assert proportions[1, 23 - 1] == 0.5
    assert proportions[2, 1 - 1] == 0.5
    assert np.allclose(proportions[:3].sum(axis=1), 1.0)


def test_a_cell_with_no_valid_pixel_is_missing_rather_than_empty() -> None:
    proportions = class_proportions(BLOCKS, NODATA)

    assert np.isnan(proportions[3]).all()


def test_the_nomenclature_nodata_class_stays_out_of_the_denominator() -> None:
    pixels = PIXELS.copy()
    pixels[0, 1] = CLC_NODATA_CODE

    proportions = class_proportions(pixels.reshape(2, 2, 2, 2), NODATA)

    assert proportions[0, 24 - 1] == pytest.approx(2 / 3)
    assert proportions[0].sum() == pytest.approx(1.0)


def test_a_value_outside_the_clc_code_range_is_rejected() -> None:
    """Codes index the count array, so an unexpected one spills into the next cell."""
    pixels = PIXELS.copy()
    pixels[0, 0] = 512  # the 3-digit code, which the cropped rasters never carry

    with pytest.raises(ValueError, match="outside CLC grid codes"):
        class_proportions(pixels.reshape(2, 2, 2, 2), NODATA)


def test_an_aggregate_holds_its_own_member_classes() -> None:
    frame = proportion_frame(class_proportions(BLOCKS, NODATA))

    # cell 0 is three quarters code 24 (312, coniferous), one quarter code 12 (211, arable)
    assert frame.loc[0, "CLC_L2_31"] == pytest.approx(0.75)
    assert frame.loc[0, "CLC_L2_21"] == pytest.approx(0.25)
    assert frame.loc[0, "CLC_L1_3"] == pytest.approx(0.75)
    # cell 1 is half code 23 (311, broad-leaved), half code 41 (512, water bodies)
    assert frame.loc[1, "CLC_L1_5"] == pytest.approx(0.5)


def test_aspect_sectors_tile_the_compass_with_north_across_zero() -> None:
    sectors = aspect_sectors()

    assert len(sectors) == 8
    assert sectors[0] == (337.5, 22.5)
    # each sector starts where the previous one ended, all the way round
    assert [start for start, _ in sectors[1:]] == [end for _, end in sectors[:-1]]
    assert sectors[-1][1] == sectors[0][0]


SMALL = GridSpec(xmin=0.0, ymax=1000.0, n_cols=2, n_rows=2, resolution_m=500, crs=CRS)


@pytest.mark.parametrize(
    ("water", "expected"),
    [
        (box(0, 500, 500, 1000), [1.0, 0.0, 0.0, 0.0]),
        (box(250, 250, 750, 750), [0.25, 0.25, 0.25, 0.25]),
        (box(9000, 9000, 9500, 9500), [0.0, 0.0, 0.0, 0.0]),
    ],
    ids=["one_full_cell", "straddling_four_cells", "outside_the_grid"],
)
def test_waterbody_fraction_measures_the_covered_share_of_each_cell(
    water: BaseGeometry, expected: list[float]
) -> None:
    assert waterbody_fraction(SMALL, water).tolist() == expected
