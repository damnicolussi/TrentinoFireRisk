"""What the retraining variants change, and that reading them changes nothing else."""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tfire.config import Config
from tfire.models.events import cell_effect_share
from tfire.models.trentino import positive_weight
from tfire.models.variants import (
    Bundle,
    CellShare,
    Variant,
    adding,
    design,
    lapsed,
    paired_median_gap,
    raw_scores,
    recency_weights,
    select,
    without_land_cover,
)


def test_recency_weights_halve_every_half_life() -> None:
    years = np.array([2014, 2004, 1994], dtype="int64")
    assert recency_weights(years, 2014, None) is None
    weights = recency_weights(years, 2014, 10.0)
    assert weights is not None
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.25])


def test_weighted_positive_weight_counts_rows_at_their_weight() -> None:
    labels = np.array([1, 1, 0, 0, 0], dtype="int8")
    assert positive_weight(labels) == pytest.approx(1.5)
    weights = np.array([1.0, 0.5, 1.0, 1.0, 0.5])
    assert positive_weight(labels, weights) == pytest.approx(2.5 / 1.5)


def _table() -> tuple[pd.DataFrame, npt.NDArray[np.int8], pd.Series]:
    years = pd.Series(np.repeat(np.arange(1984, 2025), 2))
    features = pd.DataFrame(
        {
            "a": np.arange(len(years), dtype="float32"),
            "b": np.ones(len(years), dtype="float32"),
            "c": np.zeros(len(years), dtype="float32"),
        }
    )
    labels = np.tile(np.array([1, 0], dtype="int8"), len(years) // 2)
    return features, labels, years


def test_a_first_year_drops_training_rows_and_never_holdout_rows(config: Config) -> None:
    features, labels, years = _table()
    data = design(Variant("cut", "", first_year=2000), features, labels, years, ["a", "b"], config)

    assert data.years.min() == 2000
    assert list(data.features.columns) == ["a", "b"]
    held = data.years[~data.train]
    assert held.min() == config.trentino.test_years_start
    assert len(held) == int((years >= config.trentino.test_years_start).sum())
    assert data.weights is None


def test_columns_start_from_the_reference_model_and_the_selector_narrows_them(
    config: Config,
) -> None:
    features, labels, years = _table()
    narrow = Variant("narrow", "", columns=lambda columns: [c for c in columns if c != "b"])
    data = design(narrow, features, labels, years, ["a", "b"], config)
    # "c" is in the table but not in the reference, so no variant sees it unless it asks
    assert list(data.features.columns) == ["a"]


def test_weights_follow_the_rows_a_variant_keeps(config: Config) -> None:
    features, labels, years = _table()
    data = design(
        Variant("both", "", half_life=5.0, first_year=2010), features, labels, years, ["a"], config
    )
    assert data.weights is not None
    assert len(data.weights) == len(data.labels)
    last = config.trentino.test_years_start - 1
    np.testing.assert_allclose(data.weights, 0.5 ** ((last - data.years.to_numpy("int64")) / 5.0))


class SumEstimator:
    """Scores a row by a fixed weighted sum of its columns, through a logistic."""

    def __init__(self, coefficients: dict[str, float]) -> None:
        self.coefficients = coefficients
        self.seen: list[str] = []

    def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        self.seen = list(features.columns)
        margin = sum(
            features[name].to_numpy("float64") * w for name, w in self.coefficients.items()
        )
        positive = 1 / (1 + np.exp(-np.asarray(margin)))
        return np.column_stack([1 - positive, positive])

    def fit(self, *args: object, **kwargs: object) -> SumEstimator:
        return self

    def set_params(self, **params: object) -> SumEstimator:
        return self


def test_each_bundle_sees_only_its_own_columns_in_its_own_order() -> None:
    features = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "season_summer": [1.0, 1.0], "z": [5.0, 6.0]}
    )
    wide = SumEstimator({"x": 1.0, "z": 0.5})
    narrow = SumEstimator({"z": -1.0})
    wide_columns = ["z", "x", "season_winter", "season_summer"]

    wide_scores = raw_scores(Bundle(wide, wide_columns), features)
    narrow_scores = raw_scores(Bundle(narrow, ["z"]), features)

    assert wide.seen == wide_columns
    assert narrow.seen == ["z"]
    np.testing.assert_allclose(wide_scores, 1 / (1 + np.exp(-np.array([3.5, 5.0]))))
    np.testing.assert_allclose(narrow_scores, 1 / (1 + np.exp(np.array([5.0, 6.0]))))
    # an absent season indicator is genuinely zero, as in the single-model path
    assert (select(features, wide_columns)["season_winter"] == 0).all()


def test_the_running_cell_share_matches_the_one_shot_one() -> None:
    rng = np.random.default_rng(3)
    cells = np.arange(50, dtype="int64")
    cell_effect = rng.normal(0, 1, len(cells))
    days = {date(2024, 8, d): cell_effect + rng.normal(0, 0.5, len(cells)) for d in range(1, 16)}

    running = CellShare()
    for values in days.values():
        running.add(cells, values)
    one_shot = cell_effect_share({day: 10.0**values for day, values in days.items()})
    assert running.share() == pytest.approx(one_shot, rel=1e-9)

    shifted = CellShare()
    for values in days.values():
        shifted.add(cells, 3.0 * values - 7.0)
    assert shifted.share() == pytest.approx(running.share(), rel=1e-9)


def test_the_running_cell_share_refuses_a_changed_grid() -> None:
    share = CellShare()
    share.add(np.arange(3, dtype="int64"), np.zeros(3))
    with pytest.raises(ValueError, match="different cell sets"):
        share.add(np.arange(1, 4, dtype="int64"), np.zeros(3))


def test_land_cover_selectors_drop_only_their_levels() -> None:
    columns = ["elevation_mean", "CLC_1", "CLC_44", "CLC_L2_11", "CLC_L2_52", "CLC_L1_1", "ndvi"]
    assert without_land_cover("l3")(columns) == [
        "elevation_mean",
        "CLC_L2_11",
        "CLC_L2_52",
        "CLC_L1_1",
        "ndvi",
    ]
    assert without_land_cover("l3", "l1")(columns) == [
        "elevation_mean",
        "CLC_L2_11",
        "CLC_L2_52",
        "ndvi",
    ]
    assert without_land_cover("l3", "l2", "l1")(columns) == ["elevation_mean", "ndvi"]


def test_the_paired_median_gap_is_exact_for_a_shift_and_zero_for_itself() -> None:
    rng = np.random.default_rng(5)
    reference = rng.uniform(0, 1, 101)
    days = [f"2020-01-{1 + k % 28:02d}" for k in range(101)]

    same = paired_median_gap(reference, reference.copy(), days, 200, 0)
    assert same == {"delta": 0.0, "lo": 0.0, "hi": 0.0}

    shifted = paired_median_gap(reference, reference + 0.1, days, 200, 0)
    assert shifted["delta"] == pytest.approx(0.1)
    assert shifted["lo"] == pytest.approx(0.1)
    assert shifted["hi"] == pytest.approx(0.1)


def test_the_paired_median_gap_resamples_whole_days() -> None:
    # one day carries every ignition, so each resample redraws the same rows
    reference = np.array([0.2, 0.5, 0.9])
    other = np.array([0.1, 0.8, 0.7])
    gap = paired_median_gap(reference, other, ["2020-08-01"] * 3, 50, 1)
    assert gap["lo"] == gap["hi"] == pytest.approx(0.7 - 0.5)


def test_added_features_follow_the_reference_columns_once() -> None:
    assert adding("b", "c")(["a", "b"]) == ["a", "b", "c"]


def test_the_lapse_variant_swaps_temperature_and_humidity_in_place() -> None:
    columns = ["elevation_mean", "temp_max", "temp_range", "rh_min", "fwi"]
    assert lapsed("vpd_max")(columns) == [
        "elevation_mean",
        "temp_max_lapse",
        "temp_range",
        "rh_min_lapse",
        "fwi",
        "vpd_max_lapse",
    ]


def test_the_v2_bounds_are_still_the_ones_v2_was_searched_in() -> None:
    from tfire.models.trentino import SEARCH_SPACE, SEARCH_SPACES

    assert SEARCH_SPACES["v2"] == SEARCH_SPACE
    assert SEARCH_SPACE["colsample_bytree"] == (0.4, 1.0)
    assert set(SEARCH_SPACES["v3"]) == {*SEARCH_SPACE, "colsample_bynode"}
    assert SEARCH_SPACES["v3"]["colsample_bytree"] == (0.1, 1.0)


def test_a_parameter_left_on_a_bound_is_reported() -> None:
    from tfire.models.trentino import on_the_boundary

    space = {"a": (0.4, 1.0), "b": (0.0, 10.0), "c": (1.0, 2.0)}
    assert on_the_boundary({"a": 0.4005, "b": 5.0, "c": 1.99}, space) == ["a", "c"]
    assert on_the_boundary({"a": 0.7, "b": 0.5}, space) == []


def test_only_a_variant_with_its_own_objective_or_bounds_searches() -> None:
    assert not Variant("plain", "").retunes
    assert Variant("searched", "", objective="fold_lift").retunes
    assert Variant("bounded", "", search_space="v3").retunes
