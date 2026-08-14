"""The Mesogeos schema mapping: class grouping, aspect encoding, units, and column alignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tfire.config import Config
from tfire.models.mesogeos import (
    CLC_TO_MESOGEOS,
    COMMON_FEATURES,
    LAND_COVER_CLASSES,
    MESOGEOS_SCALING,
    PRECIP_WINDOWS,
    UNMAPPED,
    aspect_components,
    from_trentino,
    load_mesogeos,
)
from tfire.sources.dem import N_ASPECT_CLASSES

N_CLC_CODES = 44

TRENTINO_LABELS = frozenset({"is_fire", "is_near_fire", "fire_id", "n_fires"})
MESOGEOS_LABELS = frozenset({"burned_areas", "ignition_points", "burned_area_has"})


def trentino_row(**overrides: float) -> pd.DataFrame:
    """One assembled sample: every column `from_trentino` reads, all zero unless overridden."""
    columns: dict[str, float] = {f"CLC_{code}": 0.0 for code in range(1, N_CLC_CODES + 1)}
    columns |= {f"aspect_{index + 1}": 0.0 for index in range(N_ASPECT_CLASSES)}
    columns |= {
        "elevation_mean": 1000.0,
        "ndvi": 0.5,
        "temp_max": 20.0,
        "rh_min": 40.0,
        "wind_speed_max": 3.0,
        "pres_max": 900.0,
        "precip_sum": 1.0,
        "precip_cum7": 5.0,
        "precip_cum15": 9.0,
        "precip_cum30": 15.0,
        "dist_roads_mean": 120.0,
        "pop_density": 8.0,
    }
    return pd.DataFrame([columns | overrides])


def write_track(path: Path, sample_ids: list[int], length: int) -> None:
    """A minimal track file: constant fields and 1 mm of rain a day."""
    rows = [
        {
            "time": f"2010-06-{index + 1:02d}",
            "time_idx": index,
            "sample": sample,
            "aspect": 90.0,
            "dem": 500.0,
            "ndvi": 0.4,
            "t2m": 300.0,
            "rh": 0.3,
            "wind_speed": 4.0,
            "sp": 100000.0,
            "tp": 0.001,
            "roads_distance": 2.0,
            "population": 10.0,
            **{f"lc_{name}": float(name == "forest") for name in LAND_COVER_CLASSES},
        }
        for sample in sample_ids
        for index in range(length)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def tracks(tmp_path: Path, config: Config) -> Config:
    """A config pointing at two synthetic track files whose sample ids deliberately collide."""
    length = config.mesogeos.track_length_days
    write_track(tmp_path / "positives.csv", [0, 1], length)
    write_track(tmp_path / "negatives.csv", [0, 1], length)
    paths = config.paths.model_copy(update={"mesogeos_raw": tmp_path})
    return config.model_copy(update={"paths": paths})


def test_every_corine_code_lands_in_exactly_one_mesogeos_class() -> None:
    """A code left out silently loses its share; the eight fractions still look like fractions."""
    mapped = [code for codes in CLC_TO_MESOGEOS.values() for code in codes]

    assert sorted(mapped) == list(range(1, N_CLC_CODES + 1))
    assert set(CLC_TO_MESOGEOS) == set(LAND_COVER_CLASSES)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(2, "lc_settlement"), (18, "lc_grassland"), (28, "lc_shrubland"), (41, "lc_water_bodies")],
    ids=["urban_fabric", "pastures", "sclerophyllous", "water_bodies"],
)
def test_land_cover_proportions_survive_the_regrouping(code: int, expected: str) -> None:
    frame = trentino_row(**{f"CLC_{code}": 0.6, "CLC_25": 0.4})

    common = from_trentino(frame).iloc[0]
    classes = [f"lc_{name}" for name in LAND_COVER_CLASSES]

    assert common[classes].sum() == pytest.approx(1.0)
    assert common[expected] == pytest.approx(0.6)
    assert common["lc_forest"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("sector", "northness", "eastness"),
    [(1, 1.0, 0.0), (3, 0.0, 1.0), (5, -1.0, 0.0), (7, 0.0, -1.0)],
    ids=["north", "east", "south", "west"],
)
def test_aspect_fractions_become_a_unit_vector(
    sector: int, northness: float, eastness: float
) -> None:
    """Sector 1 is centered on north; a rotated convention would tilt every cell in the table."""
    north, east = aspect_components(trentino_row(**{f"aspect_{sector}": 1.0}))

    assert north[0] == pytest.approx(northness, abs=1e-9)
    assert east[0] == pytest.approx(eastness, abs=1e-9)


def test_flat_and_opposing_slopes_both_cancel_to_zero() -> None:
    """Flat ground has no bearing, and neither does a cell split evenly north and south."""
    for frame in (trentino_row(), trentino_row(aspect_1=0.5, aspect_5=0.5)):
        north, east = aspect_components(frame)

        assert north[0] == pytest.approx(0.0, abs=1e-9)
        assert east[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("common", "raw", "expected"),
    [
        ("temp_max_c", 293.15, 20.0),
        ("rh_min_pct", 0.4, 40.0),
        ("pres_max_hpa", 90000.0, 900.0),
        ("precip_mm", 0.0125, 12.5),
        ("dist_roads_m", 1.5, 1500.0),
        ("elevation_m", 1000.0, 1000.0),
    ],
    ids=["kelvin", "fraction", "pascal", "metres_of_rain", "kilometres", "passthrough"],
)
def test_mesogeos_units_convert_in_the_right_direction(
    common: str, raw: float, expected: float
) -> None:
    _, scale, offset = MESOGEOS_SCALING[common]

    assert raw * scale + offset == pytest.approx(expected)


def test_the_two_track_files_keep_their_colliding_sample_ids_apart(tracks: Config) -> None:
    """Both files number from 0, so a shared id would fuse two unrelated 30-day sequences."""
    frame = load_mesogeos(tracks)

    assert len(frame) == 4
    assert frame["label"].tolist() == [1, 1, 0, 0]


def test_the_cumulative_windows_end_on_the_target_day(tracks: Config) -> None:
    """1 mm a day makes each window its own length, which an off-by-one would not."""
    row = load_mesogeos(tracks).iloc[0]

    for window in PRECIP_WINDOWS:
        assert row[f"precip_cum{window}_mm"] == pytest.approx(float(window))


def test_both_adapters_produce_the_same_columns_in_the_same_order(tracks: Config) -> None:
    """Misaligned columns hand XGBoost the wrong feature under the right name."""
    mesogeos = load_mesogeos(tracks).drop(columns=["label", "year"])

    assert list(mesogeos.columns) == list(COMMON_FEATURES)
    assert list(from_trentino(trentino_row()).columns) == list(COMMON_FEATURES)


def test_no_label_reaches_the_common_feature_vector() -> None:
    """`ignition_points` on the target day is the label; neither side's labels may leak in."""
    assert set(UNMAPPED) >= MESOGEOS_LABELS
    assert not MESOGEOS_LABELS & {source for source, _, _ in MESOGEOS_SCALING.values()}
    assert not (MESOGEOS_LABELS | TRENTINO_LABELS) & set(COMMON_FEATURES)
