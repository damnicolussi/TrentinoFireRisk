"""Timestamp builder, field normalization and time-flag tests.

All fixtures are synthetic pandas objects, no shapefile I/O, so CI stays green
without the gitignored data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tfire.config import Config
from tfire.fires import (
    Component,
    build_datetime,
    flag_near_midnight,
    flag_suspicious_default_time,
    normalize_component,
)


def _s(*values: object) -> pd.Series:
    return pd.Series(list(values))


@pytest.mark.parametrize(
    ("day", "month", "year", "hour", "minute", "expected"),
    [
        ("14", "8", "2003", "15", "30", pd.Timestamp("2003-08-14 15:30")),
        ("14", "8", "2003", None, None, pd.Timestamp("2003-08-14 00:00")),
        ("31", "2", "1999", "12", "0", None),  # 31 February
        ("abc", "8", "2003", None, None, None),
        ("", "8", "2003", None, None, None),
        ("14", "8", None, None, None, None),
        ("14", "8", "2003", "99", "0", pd.Timestamp("2003-08-14 00:00")),  # junk hour
        (14.0, 8.0, 2003.0, 15.0, 30.0, pd.Timestamp("2003-08-14 15:30")),  # DBF float
    ],
)
def test_build_datetime_cases(
    day: object, month: object, year: object, hour: object, minute: object, expected: object
) -> None:
    result = build_datetime(_s(day), _s(month), _s(year), _s(hour), _s(minute))
    if expected is None:
        assert pd.isna(result.iloc[0])
    else:
        assert result.iloc[0] == expected


def test_mixed_series_isolates_failures() -> None:
    result = build_datetime(
        _s("14", "31", "1"), _s("8", "2", "1"), _s("2003", "1999", "1984"), _s("15", "0", "0")
    )
    assert result.iloc[0] == pd.Timestamp("2003-08-14 15:00")
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pd.Timestamp("1984-01-01 00:00")


@pytest.mark.parametrize(
    ("component", "expected"),
    [(Component.DAY, 4), (Component.MONTH, 1), (Component.YEAR, 2025)],
)
def test_full_date_string_is_split_into_scalars(component: Component, expected: int) -> None:
    values, repaired = normalize_component(_s("04/01/2025"), component)
    assert values.iloc[0] == expected
    assert bool(repaired.iloc[0])


@pytest.mark.parametrize(
    ("text", "component", "expected"),
    [
        ("13:15:00", Component.HOUR, 13),
        ("13:15:00", Component.MINUTE, 15),
        ("00:10:00", Component.HOUR, 0),
    ],
)
def test_full_time_string_is_split_into_scalars(
    text: str, component: Component, expected: int
) -> None:
    values, repaired = normalize_component(_s(text), component)
    assert values.iloc[0] == expected
    assert bool(repaired.iloc[0])


@pytest.mark.parametrize(
    ("text", "expected_value", "expected_repaired"),
    [
        ("2024", 2024, False),  # bare scalar, passes through unrepaired
        ("n/a", None, False),  # junk, becomes NaN, not flagged as repaired
    ],
)
def test_non_date_string_is_not_treated_as_a_repair(
    text: str, expected_value: int | None, expected_repaired: bool
) -> None:
    values, repaired = normalize_component(_s(text), Component.YEAR)
    if expected_value is None:
        assert pd.isna(values.iloc[0])
    else:
        assert values.iloc[0] == expected_value
    assert bool(repaired.iloc[0]) == expected_repaired


def test_mixed_series_repairs_only_the_affected_rows() -> None:
    values, repaired = normalize_component(_s("2024", "04/01/2025", "1984"), Component.YEAR)
    assert list(values) == [2024, 2025, 1984]
    assert list(repaired) == [False, True, False]


def test_date_string_yields_nothing_for_a_time_component() -> None:
    """Asking for MINUTE from a date-shaped value must not invent a number."""
    values, repaired = normalize_component(_s("04/01/2025"), Component.MINUTE)
    assert pd.isna(values.iloc[0])
    assert not bool(repaired.iloc[0])


def test_repaired_row_round_trips_into_a_timestamp() -> None:
    """End to end on the real shape of a 2025 record."""
    day, _ = normalize_component(_s("4"), Component.DAY)
    month, _ = normalize_component(_s("1"), Component.MONTH)
    year, _ = normalize_component(_s("04/01/2025"), Component.YEAR)
    hour, _ = normalize_component(_s("13:15:00"), Component.HOUR)
    minute, _ = normalize_component(_s(15.0), Component.MINUTE)

    result = build_datetime(day, month, year, hour, minute)
    assert result.iloc[0] == pd.Timestamp("2025-01-04 13:15")


def test_near_midnight_flag(config: Config) -> None:
    hours = _s(22, 23, 0, 1, 2, 3, 15, None)
    assert list(flag_near_midnight(hours, config)) == [
        True,  # 22
        True,  # 23
        True,  # 0
        True,  # 1
        False,  # 2
        False,  # 3
        False,  # 15
        False,  # null hour is never flagged
    ]


def test_suspicious_default_time_flag(config: Config) -> None:
    hours = _s(0, 1, 2, 3, 22, 23, None)
    assert list(flag_suspicious_default_time(hours, config)) == [
        True,  # 0
        True,  # 1
        True,  # 2, inclusive upper bound
        False,  # 3
        False,  # 22
        False,  # 23
        False,  # null
    ]
