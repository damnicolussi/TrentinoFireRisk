"""What the training split and the design matrix are allowed to contain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.evaluation import blocked_folds, year_blocks
from tfire.features.registry import load_registry
from tfire.models.mesogeos import PROB_COLUMN
from tfire.models.trentino import SPECS, design_matrix, training_mask

YEARS = list(range(1984, 2015))


def sample_frame() -> pd.DataFrame:
    """A row per year, carrying one column of every registry role."""
    dates = pd.to_datetime([f"{year}-06-15" for year in YEARS])
    return pd.DataFrame(
        {
            "sample_id": np.arange(len(YEARS), dtype="int32"),
            "cell_id": np.arange(len(YEARS), dtype="int32"),
            "date": dates,
            "is_fire": np.tile([True, False], len(YEARS))[: len(YEARS)],
            "is_near_fire": False,
            "fire_id": pd.array(range(len(YEARS)), dtype="Int64"),
            "n_fires": np.ones(len(YEARS), dtype="int16"),
            "clc_edition": np.full(len(YEARS), 2018, dtype="int16"),
            "worldpop_year": np.full(len(YEARS), 2020, dtype="int16"),
            "veg_composite_date": dates,
            "natura2000_fraction": np.zeros(len(YEARS), dtype="float32"),
            "elevation_mean": np.linspace(200, 3000, len(YEARS), dtype="float32"),
            "fwi": np.linspace(0, 40, len(YEARS), dtype="float32"),
            "ndvi": np.linspace(-1, 1, len(YEARS), dtype="float32"),
            "month": np.full(len(YEARS), 6, dtype="int8"),
            "season": pd.Categorical(
                ["summer"] * len(YEARS), categories=["winter", "spring", "summer", "autumn"]
            ),
            PROB_COLUMN: np.linspace(0, 1, len(YEARS), dtype="float32"),
        }
    )


@pytest.mark.parametrize("n_folds", [2, 5, 7])
def test_folds_partition_the_years_without_sharing_one(n_folds: int) -> None:
    years = pd.Series(YEARS * 3)
    blocks = year_blocks(years, n_folds)

    assert sorted(np.concatenate(blocks).tolist()) == sorted(years.unique().tolist())
    for block in blocks:
        assert block.tolist() == list(range(block[0], block[-1] + 1))

    for train_index, validation_index in blocked_folds(years, n_folds):
        held = set(years.iloc[validation_index])
        assert held
        assert not held & set(years.iloc[train_index])
        assert len(train_index) + len(validation_index) == len(years)


def test_year_blocks_rejects_more_folds_than_years() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        year_blocks(pd.Series([2020, 2021]), 3)


def test_design_matrix_carries_features_only() -> None:
    frame = sample_frame()
    features, labels, years = design_matrix(frame, load_registry())

    forbidden = {
        "sample_id",
        "cell_id",
        "date",
        "is_fire",
        "is_near_fire",
        "fire_id",
        "n_fires",
        "clc_edition",
        "worldpop_year",
        "veg_composite_date",
        "natura2000_fraction",
    }
    assert not forbidden & set(features.columns)
    assert {"elevation_mean", "fwi", "ndvi", "month", PROB_COLUMN} <= set(features.columns)

    # every declared season, so a fold missing one still produces the same columns
    assert [name for name in features.columns if name.startswith("season_")] == [
        "season_winter",
        "season_spring",
        "season_summer",
        "season_autumn",
    ]
    assert labels.tolist() == frame["is_fire"].astype("int8").tolist()
    assert years.tolist() == YEARS


def test_training_mask_excludes_the_held_out_years(config: Config) -> None:
    years = pd.Series(range(1984, 2025))
    mask = training_mask(years, config)

    assert years[mask].max() == config.trentino.test_years_start - 1
    assert years[~mask].min() == config.trentino.test_years_start


def test_training_mask_rejects_a_split_that_holds_nothing_out(config: Config) -> None:
    with pytest.raises(ValueError, match="splits nothing off"):
        training_mask(pd.Series(range(1984, 2000)), config)


def test_model_column_selectors() -> None:
    features, _, _ = design_matrix(sample_frame(), load_registry())
    names = list(features.columns)

    assert SPECS["fwi_only"].columns(names) == ["fwi"]
    assert set(names) - set(SPECS["xgboost_no_stacking"].columns(names)) == {PROB_COLUMN}
    assert SPECS["xgboost"].columns(names) == names


def test_fwi_only_refuses_a_matrix_without_fwi() -> None:
    with pytest.raises(ValueError, match="not in the design matrix"):
        SPECS["fwi_only"].columns(["elevation_mean", "ndvi"])
