"""Canadian Fire Weather Index System indices, computed on the ERA5-Land backbone."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.features.meteo import NOON_COLUMNS
from tfire.sources.era5land import read_lattice

logger = logging.getLogger(__name__)

# the order xclim returns them in
_XCLIM_ORDER: Final = ("dc", "dmc", "ffmc", "isi", "bui", "fwi")

INDEX_NAMES: Final = ("fwi", "ffmc", "dmc", "dc", "isi", "bui")

_SPRING_MONTHS: Final = (3, 4)

# units of the four noon inputs as `meteo_daily.parquet` stores them; xclim converts from here
_INPUT_UNITS: Final[dict[str, str]] = {
    "temp_noon": "degC",
    "precip_noon24": "mm/day",
    "wind_speed_noon": "m s-1",
    "rh_noon": "%",
}


def compute_fwi(meteo: pd.DataFrame, latitudes: npt.NDArray[np.float64]) -> pd.DataFrame:
    """The six indices from the noon columns of a daily table."""
    import xarray as xr
    from xclim.indices import fire

    wide = {
        name: meteo.pivot(index="date", columns="era5_id", values=name) for name in NOON_COLUMNS
    }
    dates = wide["temp_noon"].index.to_numpy()
    ids = wide["temp_noon"].columns.to_numpy()
    if latitudes.size != ids.size:
        raise ValueError(f"{latitudes.size} latitudes for {ids.size} backbone cells")

    def series(name: str) -> xr.DataArray:
        return xr.DataArray(
            wide[name].to_numpy().astype(np.float64),
            dims=("time", "era5_id"),
            coords={"time": dates, "era5_id": ids},
            attrs={"units": _INPUT_UNITS[name]},
        )

    outputs = fire.cffwis_indices(
        tas=series("temp_noon"),
        pr=series("precip_noon24"),
        sfcWind=series("wind_speed_noon"),
        hurs=series("rh_noon"),
        lat=xr.DataArray(
            latitudes, dims=("era5_id",), coords={"era5_id": ids}, attrs={"units": "degrees_north"}
        ),
        season_method=None,
        overwintering=False,
        initial_start_up=True,
    )

    frame = pd.DataFrame(
        {
            "era5_id": np.tile(ids, len(dates)).astype("int32"),
            "date": np.repeat(dates, len(ids)).astype("datetime64[us]"),
        }
    )
    for name, output in zip(_XCLIM_ORDER, outputs, strict=True):
        frame[name] = output.transpose("time", "era5_id").to_numpy().reshape(-1).astype("float32")

    return frame[["era5_id", "date", *INDEX_NAMES]]


def validate_fwi(fwi: pd.DataFrame, config: Config) -> None:
    """Check the acceptance criteria, logging loudly."""
    ceilings = {
        "fwi": config.fwi.max_fwi,
        "ffmc": config.fwi.max_ffmc,
        "dmc": config.fwi.max_dmc,
        "dc": config.fwi.max_dc,
        "isi": config.fwi.max_isi,
        "bui": config.fwi.max_bui,
    }
    for name, ceiling in ceilings.items():
        values = fwi[name]
        missing = int(values.isna().sum())
        if missing:
            logger.error("%s: %d missing value(s)", name, missing)

        outside = int(((values < 0) | (values > ceiling)).sum())
        if outside:
            logger.error(
                "%s: %d value(s) outside [0, %g], range is [%g, %g]",
                name,
                outside,
                ceiling,
                values.min(),
                values.max(),
            )
        else:
            logger.info("%s: [%.1f, %.1f]", name, values.min(), values.max())

    peak = float(province_mean_fwi(fwi).max())
    if peak > config.fwi.max_mean_fwi:
        logger.error(
            "province-wide daily mean FWI peaks at %.1f, above the ceiling of %g",
            peak,
            config.fwi.max_mean_fwi,
        )
    else:
        logger.info("province-wide daily mean FWI peaks at %.1f", peak)

    share = spring_reset_share(fwi, config.fwi.spring_reset_dc)
    if share < config.fwi.min_spring_reset_share:
        logger.error(
            "DC falls below %g in March or April in only %.1f%% of cell-years, "
            "against the %.1f%% expected of a code that resets on winter rain",
            config.fwi.spring_reset_dc,
            100 * share,
            100 * config.fwi.min_spring_reset_share,
        )
    else:
        logger.info(
            "DC falls below %g by April in %.1f%% of cell-years",
            config.fwi.spring_reset_dc,
            100 * share,
        )


def province_mean_fwi(fwi: pd.DataFrame) -> pd.Series:
    """FWI averaged over the backbone, one value per day."""
    mean: pd.Series = fwi.groupby("date")["fwi"].mean()
    return mean


def spring_reset_share(fwi: pd.DataFrame, ceiling: float) -> float:
    """Fraction of cell-years whose DC drops under `ceiling` in March or April.

    The codes run year-round with no seasonal mask, so nothing resets DC by the calendar.
    What resets it here is winter rain, and this is the measurement of that: a DC that
    integrated without ever being wetted down would stay above the start-up value.
    """
    spring = fwi.loc[fwi["date"].dt.month.isin(_SPRING_MONTHS)]
    if spring.empty:
        raise ValueError("The table carries no March or April day to check the DC reset on")

    lowest = spring.groupby([spring["date"].dt.year, spring["era5_id"]])["dc"].min()
    return float((lowest < ceiling).mean())


def extract_fwi(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `fwi.parquet` from the daily backbone table."""
    out = config.path(config.paths.fwi_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    source = config.path(config.paths.meteo_out)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found; run `tfire extract-features --category meteo` first"
        )

    meteo = pd.read_parquet(source, columns=["era5_id", "date", *NOON_COLUMNS])
    fwi = compute_fwi(meteo, read_lattice(config).cell_latitudes())
    validate_fwi(fwi, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    fwi.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(fwi), fwi.shape[1], out)
    return fwi
