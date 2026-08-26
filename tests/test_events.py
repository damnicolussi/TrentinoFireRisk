"""Event-based verification: where recorded ignitions landed in their own day's map."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.models import events
from tfire.models.danger import CLASS_KEYS, DangerClasses

# five cells, two days, probabilities chosen so every rank is a distinct fifth
_SCORES: dict[date, list[float]] = {
    date(2019, 3, 4): [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
    date(2019, 8, 12): [9e-4, 1e-6, 2e-6, 3e-6, 4e-6],
}

# the ignition cell on each day: last on the quiet day, first on the loud one
_IGNITIONS = {date(2019, 3, 4): 4, date(2019, 8, 12): 0}


@pytest.fixture
def classes() -> DangerClasses:
    return DangerClasses(
        breaks=[1e-5, 1e-4, 3e-4, 6e-4],
        percentiles=[90.0, 99.0, 99.9, 99.99],
        class_keys=list(CLASS_KEYS),
        reference="test",
        reference_years=[2015, 2024],
        rows=len(_SCORES) * 5,
        model_version="v1",
        config_sha256="0" * 64,
    )


class FakeScorer:
    """`GridScorer.day` over the fixture, same three-tuple and the same pandas ranking."""

    def __init__(self, config: Config, days: list[date], today: date | None = None) -> None:
        self.days = days

    def day(self, day: date) -> tuple[pd.DataFrame, Any, Any]:
        probability = np.asarray(_SCORES[day], dtype="float64")
        frame = pd.DataFrame({"cell_id": np.arange(len(probability), dtype="int32")})
        rank = pd.Series(probability).rank(pct=True).to_numpy()
        return frame, probability, rank


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, config: Config) -> Config:
    """A config whose event list and scorer are the fixture, not the build tree."""
    import tfire.inference

    monkeypatch.setattr(tfire.inference, "GridScorer", FakeScorer)
    monkeypatch.setattr(
        events,
        "ignition_events",
        lambda _config: pd.DataFrame(
            {
                "cell_id": [_IGNITIONS[day] for day in sorted(_SCORES)],
                "date": [pd.Timestamp(day) for day in sorted(_SCORES)],
                "fire_id": [1, 2],
                "area_ha": [0.1, 12.0],
                "season": ["spring", "summer"],
                "month": [3, 8],
            }
        ),
    )
    monkeypatch.setattr(events, "reference_days", lambda _config: sorted(_SCORES))
    monkeypatch.setattr(events, "_season_window", lambda _config: sorted(_SCORES))
    return config


def test_an_ignition_is_scored_at_its_own_cells_rank_within_its_own_day(
    wired: Config, classes: DangerClasses
) -> None:
    """A join on the wrong key still returns a number, and it looks like a percentile."""
    report = events.verify_events(wired, classes)

    # 5e-6 is the highest of five on 4 March, 9e-4 the highest of five on 12 August
    assert report["overall"]["events"] == 2
    assert report["overall"]["median_percentile"] == pytest.approx(1.0)
    assert report["overall"]["share_at_or_above_90"] == 1.0
    assert report["overall"]["share_at_or_above_99"] == 1.0

    # the quiet day's top cell is still below the first break, the loud day's is past the last
    assert report["class_distribution"] == {
        "very_low": 1,
        "low": 0,
        "moderate": 0,
        "high": 0,
        "very_high": 1,
    }
    assert set(report["by_season"]) == {"spring", "summer"}
    assert report["days_scored"] == 2


def test_a_ranking_that_is_the_same_picture_every_day_reads_as_a_cell_effect() -> None:
    """The number that says whether the weather moves the map at all."""
    frozen = {
        date(2019, 1, 1): np.array([1e-6, 1e-4]),
        date(2019, 1, 2): np.array([1e-6, 1e-4]),
    }
    assert events.cell_effect_share(frozen) == pytest.approx(1.0)

    # the same two cells, swapped day to day: nothing is fixed about the cell
    swapped = {
        date(2019, 1, 1): np.array([1e-6, 1e-4]),
        date(2019, 1, 2): np.array([1e-4, 1e-6]),
    }
    assert events.cell_effect_share(swapped) == pytest.approx(0.0)
