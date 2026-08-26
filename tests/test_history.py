"""The causal ignition-density feature: what each year is allowed to have seen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.features import history
from tfire.grid import GridSpec

SPEC = GridSpec(xmin=0.0, ymax=5000.0, n_cols=5, n_rows=5, resolution_m=1000, crs="EPSG:25832")

BURNED = 12


def test_the_kernel_puts_its_mass_on_the_cell_the_fire_was_in() -> None:
    """A sign or an axis swapped in the distance would move the field somewhere plausible."""
    x, y = SPEC.cell_center([BURNED])
    density = history.kernel_density(SPEC, x, y, bandwidth_m=1000.0)

    assert int(np.argmax(density)) == BURNED
    # it falls off with distance rather than staying flat, and it is symmetric about the event
    assert density[BURNED] > density[7] > density[2]
    assert density[7] == pytest.approx(density[17])
    assert density[11] == pytest.approx(density[13])


def test_no_event_leaves_an_empty_field_rather_than_a_division() -> None:
    empty = np.array([], dtype="float64")
    assert history.kernel_density(SPEC, empty, empty, bandwidth_m=1000.0).tolist() == [0.0] * 25


@pytest.fixture
def one_fire(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config whose whole cadastre is a single 2000 event in the middle of a 5x5 grid."""

    def fake_load_grid(_config: Config) -> tuple[GridSpec, pd.DataFrame]:
        frame = pd.DataFrame(
            {
                "cell_id": np.arange(SPEC.n_cells, dtype="int32"),
                "is_trentino": np.ones(SPEC.n_cells, dtype=bool),
            }
        )
        return SPEC, frame

    monkeypatch.setattr(history, "load_grid", fake_load_grid)

    x, y = SPEC.cell_center([BURNED])
    fires = tmp_path / "fires.parquet"
    pd.DataFrame(
        {
            "fire_id": [1],
            "x": x,
            "y": y,
            "ignition_date": [pd.Timestamp("2000-07-01")],
            "end_datetime": [pd.Timestamp("2000-07-01")],
        }
    ).to_parquet(fires, index=False)

    return config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "fires_out": fires,
                    "fire_history_out": tmp_path / "fire_history.parquet",
                }
            ),
            "history": config.history.model_copy(update={"bandwidth_m": 1000.0}),
        }
    )


def test_a_fire_never_contributes_to_the_density_of_its_own_year(one_fire: Config) -> None:
    """Leakage here inflates every metric downstream and none of the numbers look wrong."""
    table = history.extract_history(one_fire, force=True)
    burned = table[table["cell_id"] == BURNED].set_index("year")[history.DENSITY_COLUMN]
    window = one_fire.history.window_years

    # the year of the fire, and the year before it, know nothing about it
    assert burned.loc[1999] == 0.0
    assert burned.loc[2000] == 0.0
    # the year after does, and so does the last year of the trailing window
    assert burned.loc[2001] > 0.0
    assert burned.loc[2000 + window] > 0.0
    # past the window it has aged out again
    assert burned.loc[2001 + window] == 0.0


def test_the_table_reaches_one_year_past_the_record_so_serving_has_a_value(
    config: Config,
) -> None:
    """A 2026 assembly finding no density would fail the feature contract, not fall back."""
    years = history._years(config)
    end = config.meteo.extension_end or config.date_range.end
    assert years[0] == config.date_range.start.year
    assert years[-1] == max(config.date_range.end.year, end.year) + 1
