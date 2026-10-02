"""Single-fire verification and the extreme-risk episodes: where a fire lands, what joins what."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from tfire.grid import GridSpec
from tfire.models.cases import cells_within, locate
from tfire.models.episodes import label_episodes

_CRS = "EPSG:25832"

# 10 x 10 cells of 500 m around Trento
_SPEC = GridSpec(xmin=660_000.0, ymax=5_110_000.0, n_cols=10, n_rows=10, resolution_m=500, crs=_CRS)


def test_a_fire_given_in_degrees_lands_on_the_cell_whose_center_it_is() -> None:
    """A swapped lat/lon or axis order still returns a cell, just one in another valley."""
    cell = 43
    x, y = _SPEC.cell_center(cell)
    lon, lat = Transformer.from_crs(_CRS, "EPSG:4326", always_xy=True).transform(x, y)

    _, _, found = locate(_SPEC, _CRS, lat, lon)

    assert found == cell


def test_the_uncertainty_radius_takes_every_cell_centered_inside_it() -> None:
    active = np.arange(_SPEC.n_cells, dtype="int64")
    x, y = (float(value) for value in _SPEC.cell_center(55))

    # 500 m reaches the four edge neighbors, 1000 m adds the diagonals and the next ring's axes
    assert len(cells_within(_SPEC, active, x, y, 0.0)) == 1
    assert len(cells_within(_SPEC, active, x, y, 500.0)) == 5
    assert len(cells_within(_SPEC, active, x, y, 1000.0)) == 13

    # a point off center on an inactive neighborhood still keeps its own cell
    alone = np.array([55], dtype="int64")
    assert cells_within(_SPEC, alone, x + 200.0, y, 100.0).tolist() == [55]


@pytest.mark.parametrize(
    ("cells", "episodes"),
    [
        # diagonal neighbors on the same day
        ([(0, 2, 2), (0, 3, 3)], 1),
        # the same cell two days running
        ([(0, 2, 2), (1, 2, 2)], 1),
        # a neighbor on the next day
        ([(0, 2, 2), (1, 3, 2)], 1),
        # a one-day gap splits
        ([(0, 2, 2), (2, 2, 2)], 2),
        # two cells apart on one day
        ([(0, 2, 2), (0, 2, 4)], 2),
    ],
)
def test_extreme_cell_days_join_in_space_and_in_time(
    cells: list[tuple[int, int, int]], episodes: int
) -> None:
    mask = np.zeros((3, 6, 6), dtype=bool)
    for day, row, col in cells:
        mask[day, row, col] = True

    _, count = label_episodes(mask)

    assert count == episodes


def test_an_italian_date_is_read_day_first_and_a_decimal_comma_as_a_point() -> None:
    """Month first would turn 11 January into 1 November without raising."""
    from tfire.models.cases import Case

    case = Case.model_validate(
        {
            "id": "x",
            "date": "11/01/2026",
            "lat": "46.0",
            "lon": "11.0",
            "uncertainty_m": "500",
            "coord_source": "cell",
            "place": "p",
            "area_ha": "0,6",
            "cause": "unknown",
        }
    )

    assert str(case.date) == "2026-01-11"
    assert case.area_ha == 0.6
