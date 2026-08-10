"""FWI indices: seasonal continuity of the drought code and plausible ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfire.config import Config
from tfire.features.fwi import INDEX_NAMES, compute_fwi

N_CELLS = 2
LATITUDES = np.full(N_CELLS, 46.0)

# what the codes are initialized to on a cold start
_STARTUP_DC = 15.0


def daily_table(
    n_years: int = 3, start: str = "2001-01-01", rain_mm: float | None = None
) -> pd.DataFrame:
    """A backbone table shaped like `meteo_daily.parquet`, on a plain seasonal cycle."""
    dates = pd.date_range(start, periods=n_years * 365, freq="D")
    season = np.sin((dates.dayofyear - 100) * 2 * np.pi / 365.0)
    rng = np.random.default_rng(0)

    rain = rng.gamma(0.4, 3.0, len(dates)) if rain_mm is None else np.full(len(dates), rain_mm)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), N_CELLS),
            "era5_id": np.tile(np.arange(N_CELLS, dtype="int32"), len(dates)),
            "temp_noon": np.repeat(10.0 + 15.0 * season, N_CELLS),
            "rh_noon": np.repeat(60.0 - 20.0 * season, N_CELLS),
            "wind_speed_noon": 3.0,
            "precip_noon24": np.repeat(rain, N_CELLS),
        }
    )
    return frame


def test_indices_stay_in_plausible_ranges(config: Config) -> None:
    fwi = compute_fwi(daily_table(), LATITUDES)

    assert list(fwi.columns) == ["era5_id", "date", *INDEX_NAMES]
    assert not fwi[list(INDEX_NAMES)].isna().to_numpy().any()

    ceilings = {
        "fwi": config.fwi.max_fwi,
        "ffmc": config.fwi.max_ffmc,
        "dmc": config.fwi.max_dmc,
        "dc": config.fwi.max_dc,
        "isi": config.fwi.max_isi,
        "bui": config.fwi.max_bui,
    }
    for name, ceiling in ceilings.items():
        assert fwi[name].min() >= 0.0
        assert fwi[name].max() <= ceiling


def test_drought_code_carries_across_the_year_boundary() -> None:
    fwi = compute_fwi(daily_table(), LATITUDES).sort_values(["era5_id", "date"])

    step = fwi.groupby("era5_id")["dc"].diff().abs()
    boundary = fwi["date"].dt.dayofyear == 1

    # a seasonal restart would drop DC back to its start-up value in one day
    assert step[boundary].max() <= step[~boundary].max()

    # the opening day starts cold, which is what the spin-up year absorbs; every
    # January after it inherits the previous season
    later = boundary & (fwi["era5_id"] == 0) & (fwi["date"] > fwi["date"].min())
    assert (fwi.loc[later, "dc"] > _STARTUP_DC).all()


def test_drought_code_climbs_through_a_rainless_stretch() -> None:
    fwi = compute_fwi(daily_table(rain_mm=0.0), LATITUDES)

    summer = fwi.loc[
        (fwi["era5_id"] == 0) & (fwi["date"] >= "2002-06-01") & (fwi["date"] <= "2002-08-31"),
        "dc",
    ]
    assert (summer.diff().dropna() > 0).all()


def test_wetting_rain_pulls_the_drought_code_down() -> None:
    dry = compute_fwi(daily_table(rain_mm=0.0), LATITUDES)
    wet = compute_fwi(daily_table(rain_mm=20.0), LATITUDES)

    assert wet["dc"].max() < dry["dc"].max()
    assert wet["fwi"].mean() < dry["fwi"].mean()


@pytest.mark.parametrize("n_latitudes", [1, 3], ids=["too few", "too many"])
def test_latitudes_must_match_the_backbone(n_latitudes: int) -> None:
    with pytest.raises(ValueError, match="backbone cells"):
        compute_fwi(daily_table(n_years=1), np.full(n_latitudes, 46.0))
