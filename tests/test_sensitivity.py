"""The label shift the near-midnight variant applies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfire.config import Config
from tfire.features.registry import Registry, load_registry
from tfire.models.trentino import design_matrix
from tfire.sensitivity import VARIANTS, resolve, shifted_samples


def samples() -> pd.DataFrame:
    """Four positives around midnight, one at noon, and a negative sitting where one lands."""
    return pd.DataFrame(
        {
            "sample_id": np.arange(6, dtype="int32"),
            "cell_id": np.array([10, 11, 12, 13, 14, 11], dtype="int32"),
            "date": pd.to_datetime(
                [
                    "1990-06-15",  # 23:00, moves to the 16th
                    "1990-06-15",  # 00:30, moves to the 14th
                    "1990-06-15",  # 12:00, stays
                    "1984-01-01",  # 01:00, would move before the record
                    "2024-12-31",  # 22:30, would move past it
                    "1990-06-14",  # the negative the second positive lands on
                ]
            ),
            "is_fire": [True, True, True, True, True, False],
            "fire_id": pd.array([1, 2, 3, 4, 5, pd.NA], dtype="Int64"),
        }
    )


def fires() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fire_id": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "start_hour": [23, 0, 12, 1, 22],
        }
    )


def test_near_midnight_shift_moves_each_hour_to_its_own_side(config: Config) -> None:
    shifted = shifted_samples(samples(), fires(), config).set_index("sample_id")

    assert shifted.loc[0, "date"] == pd.Timestamp("1990-06-16")
    assert shifted.loc[1, "date"] == pd.Timestamp("1990-06-14")
    assert shifted.loc[2, "date"] == pd.Timestamp("1990-06-15")
    assert 3 not in shifted.index
    assert 4 not in shifted.index


def test_near_midnight_collisions_keep_the_positive(config: Config) -> None:
    shifted = shifted_samples(samples(), fires(), config)

    landed = shifted.loc[(shifted["cell_id"] == 11) & (shifted["date"] == "1990-06-14")]
    assert len(landed) == 1
    assert bool(landed["is_fire"].iloc[0])
    assert not shifted.duplicated(["cell_id", "date"]).any()


def test_resolve_defaults_to_every_variant() -> None:
    assert resolve(None) == list(VARIANTS)
    assert resolve(["all"]) == list(VARIANTS)
    assert resolve(["negative_ratio_5"]) == ["negative_ratio_5"]


def test_dropping_a_block_removes_exactly_its_columns() -> None:
    """A category name that matches nothing would report the baseline as an ablation."""
    registry = load_registry()
    frame = pd.DataFrame(
        {"is_fire": [0, 1], "date": pd.to_datetime(["1990-01-01", "1990-01-02"])}
        | {spec.name: [0.0, 1.0] for spec in registry.features if spec.dtype != "category"}
    )

    full, _, _ = design_matrix(frame, registry)
    for block in {spec.category for spec in registry.present(frame)}:
        kept = [spec for spec in registry.specs if spec.category != block]
        reduced, _, _ = design_matrix(frame, Registry(kept, registry.categories))
        dropped = set(full.columns) - set(reduced.columns)
        assert dropped == {spec.name for spec in registry.present(frame) if spec.category == block}
        assert set(reduced.columns) < set(full.columns)
