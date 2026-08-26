"""Quantile mapping of the remotely served fields onto the backbone distribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.sources import bias


def ladder_table(cells: list[int], months: list[int], factor: float) -> pd.DataFrame:
    """A map whose served distribution is the cached one stretched by `factor`."""
    rungs = np.linspace(0.0, 100.0, bias.RUNGS)
    cached = np.linspace(0.0, 10.0, bias.RUNGS)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "era5_id": cell,
                    "month": month,
                    "column": "wind_speed_mean",
                    "rung": rungs,
                    "cached": cached,
                    "served": cached * factor,
                }
            )
            for cell in cells
            for month in months
        ],
        ignore_index=True,
    )


def tweak(config: Config, **bias_fields: object) -> Config:
    return config.model_copy(update={"bias": config.bias.model_copy(update=bias_fields)})


def test_a_served_field_comes_back_on_the_backbones_distribution(config: Config) -> None:
    """A correction that lands on the wrong cell or month is a plausible wrong wind speed."""
    moved = tweak(config, columns=["wind_speed_mean"])
    table = ladder_table(cells=[0, 1], months=[6], factor=2.0)

    served = pd.DataFrame(
        {
            "era5_id": [0, 0, 1, 1],
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"] * 2),
            "wind_speed_mean": [2.0, 8.0, 4.0, 10.0],
        }
    )
    corrected = bias.correct(served, table, moved)

    # the served ladder is exactly twice the cached one, so every value halves
    assert corrected["wind_speed_mean"].tolist() == pytest.approx([1.0, 4.0, 2.0, 5.0])
    # and nothing else on the frame moved
    assert corrected["era5_id"].tolist() == served["era5_id"].tolist()


def test_a_cell_month_the_map_never_saw_is_left_alone(config: Config) -> None:
    moved = tweak(config, columns=["wind_speed_mean"])
    table = ladder_table(cells=[0], months=[6], factor=2.0)

    served = pd.DataFrame(
        {
            "era5_id": [0, 0],
            "date": pd.to_datetime(["2026-06-01", "2026-07-01"]),
            "wind_speed_mean": [2.0, 2.0],
        }
    )
    corrected = bias.correct(served, table, moved)
    assert corrected["wind_speed_mean"].tolist() == pytest.approx([1.0, 2.0])


def test_a_value_past_the_fitted_range_stays_extreme(config: Config) -> None:
    """Wrapping an unprecedented wind back into the middle of the record would hide it."""
    moved = tweak(config, columns=["wind_speed_mean"])
    table = ladder_table(cells=[0], months=[6], factor=2.0)

    served = pd.DataFrame(
        {
            "era5_id": [0, 0],
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "wind_speed_mean": [-5.0, 500.0],
        }
    )
    corrected = bias.correct(served, table, moved)
    assert corrected["wind_speed_mean"].tolist() == pytest.approx([0.0, 10.0])


def test_a_corrected_daily_maximum_never_falls_under_its_own_mean(config: Config) -> None:
    """Each column is mapped on its own ladder, so the two can cross without a guard."""
    moved = tweak(config, columns=["wind_speed_mean", "wind_speed_max"])
    table = pd.concat(
        [
            ladder_table(cells=[0], months=[6], factor=1.0),
            ladder_table(cells=[0], months=[6], factor=4.0).assign(column="wind_speed_max"),
        ],
        ignore_index=True,
    )

    served = pd.DataFrame(
        {
            "era5_id": [0],
            "date": pd.to_datetime(["2026-06-01"]),
            "wind_speed_mean": [6.0],
            "wind_speed_max": [8.0],
        }
    )
    corrected = bias.correct(served, table, moved)

    assert corrected["wind_speed_mean"].iloc[0] == pytest.approx(6.0)
    assert corrected["wind_speed_max"].iloc[0] >= corrected["wind_speed_mean"].iloc[0]


def test_a_remote_map_scored_under_a_different_correction_is_stale(
    config: Config, tmp_path: Path
) -> None:
    """Refitting the wind map moves every remote probability and touches nothing else."""
    from tfire.sources.bias import bias_fingerprint

    paths = config.paths.model_copy(update={"bias_map_out": tmp_path / "bias.parquet"})
    moved = config.model_copy(update={"paths": paths})
    assert bias_fingerprint(moved) is None

    ladder_table(cells=[0], months=[6], factor=2.0).to_parquet(moved.paths.bias_map_out)
    first = bias_fingerprint(moved)
    assert first is not None

    # a refit with different content has to read as a different correction
    ladder_table(cells=[0, 1], months=[6, 7], factor=3.0).to_parquet(moved.paths.bias_map_out)
    assert bias_fingerprint(moved) != first
