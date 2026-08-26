"""Quantile mapping that puts remotely served fields back on the backbone's distribution."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config

if TYPE_CHECKING:
    from tfire.sources.era5land import Lattice

logger = logging.getLogger(__name__)

# the ladder both distributions are read on. 101 rungs resolves a percentile, which is finer
# than the difference between the two sources anywhere but the extreme tails.
RUNGS: Final = 101

# days taken from each month of the reference period. Ten of them over five years is fifty
# observations behind every cell-month, which is what the ladder is interpolated from.
_SAMPLE_DAYS: Final = 10

# seconds between spans. Open-Meteo's minutely allowance is generous for a handful of requests
# and this asks for sixty spans in a row, so the fit paces itself instead of leaning on the
# retry backoff, which turns a rate limit into an hour of exponential waiting.
_PACE_SECONDS: Final = 2

# consecutive refusals after which the fetching stops. The endpoint rations by the hour, so once
# it has said no this many times in a row it will keep saying no for a while, and every further
# span costs a full retry ladder to learn nothing. Whatever is already cached still gets fitted.
_GIVE_UP_AFTER: Final = 3


def _ladder() -> npt.NDArray[np.float64]:
    return np.linspace(0.0, 100.0, RUNGS)


def _served_span(config: Config, start: date, end: date) -> pd.DataFrame:
    """The reference period through the operational path, aggregated exactly as training was.

    Open-Meteo prices an archive call by hours x locations x variables, and over 187 cells a
    span of any length is expensive enough to earn a 429. Two things keep the ask small. The
    corrected columns are all same-day statistics, so unlike the serving path this needs no
    lead window for the lag features. And a quantile map wants the shape of a distribution
    rather than every day in it, so it takes the opening `_SAMPLE_DAYS` of each month.
    """
    from tfire.features.meteo import aggregate_daily, to_frame
    from tfire.sources import forecast
    from tfire.sources.era5land import read_lattice

    lattice: Lattice = read_lattice(config)

    spans = _chunks(start, end)
    parts = []
    refused = []
    consecutive = 0
    for index, (first, last) in enumerate(spans):
        if consecutive >= _GIVE_UP_AFTER:
            refused.append(first)
            continue
        if index:
            time.sleep(_PACE_SECONDS)
        try:
            fields, _ = forecast.fetch_span(config, lattice, first, last)
        except forecast.ForecastError as error:
            # a five-year fetch against a rate-limited free endpoint cannot be all or nothing:
            # every span already on disk still counts, and a later run fills in the rest
            refused.append(first)
            consecutive += 1
            logger.warning("Skipping %s to %s: %s", first, last, error)
            if consecutive == _GIVE_UP_AFTER:
                logger.warning("Refused %d times running, fitting on what is cached", consecutive)
            continue
        consecutive = 0

        dates, columns = aggregate_daily(fields, config)
        frame = to_frame(dates, columns, lattice)
        parts.append(frame[frame["date"].between(pd.Timestamp(first), pd.Timestamp(last))])
        logger.info("Served %s to %s: %d cell-day(s)", first, last, len(parts[-1]))

    if refused:
        logger.warning(
            "%d of %d span(s) were refused; run again to extend the fit", len(refused), len(spans)
        )
    if not parts:
        raise forecast.ForecastError("No span could be fetched, so there is nothing to fit on")
    return pd.concat(parts, ignore_index=True)


def _chunks(start: date, end: date) -> list[tuple[date, date]]:
    """The opening days of every month between `start` and `end`."""
    spans = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            first = date(year, month, 1)
            last = first + timedelta(days=_SAMPLE_DAYS - 1)
            if last >= start and first <= end:
                spans.append((max(first, start), min(last, end)))
    return spans


def fit_bias_map(config: Config, force: bool = False) -> pd.DataFrame:
    """Per cell, per month, the two distributions of every corrected column, on a shared ladder.

    Fitted on the overlap where both sources cover the same days, which is the only place the
    two can be compared at all. Stored as quantiles rather than a fitted shape: the difference
    is not a constant factor, it is largest at the calm end where the backbone sits near zero.
    """
    out = config.path(config.paths.bias_map_out)
    if out.is_file() and not force:
        logger.info("%s already exists, use --force to refit", out)
        return pd.read_parquet(out)

    start = date(config.bias.reference_years[0], 1, 1)
    end = date(config.bias.reference_years[1], 12, 31)
    columns = list(config.bias.columns)

    cached = pd.read_parquet(
        config.path(config.paths.meteo_out), columns=["era5_id", "date", *columns]
    )
    cached = cached[cached["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    served = _served_span(config, start, end)

    ladder = _ladder()
    rows = []
    for column in columns:
        for (era5_id, month), part in cached.groupby([cached["era5_id"], cached["date"].dt.month]):
            reference = np.percentile(part[column].to_numpy(dtype="float64"), ladder)
            rows.append(
                pd.DataFrame(
                    {
                        "era5_id": era5_id,
                        "month": month,
                        "column": column,
                        "rung": ladder,
                        "cached": reference,
                    }
                )
            )
    reference_table = pd.concat(rows, ignore_index=True)

    rows = []
    for column in columns:
        for (era5_id, month), part in served.groupby([served["era5_id"], served["date"].dt.month]):
            rows.append(
                pd.DataFrame(
                    {
                        "era5_id": era5_id,
                        "month": month,
                        "column": column,
                        "rung": ladder,
                        "served": np.percentile(part[column].to_numpy(dtype="float64"), ladder),
                    }
                )
            )
    served_table = pd.concat(rows, ignore_index=True)

    table = reference_table.merge(
        served_table, on=["era5_id", "month", "column", "rung"], how="inner", validate="1:1"
    )

    covered = set(table["month"].unique())
    if covered != set(range(1, 13)):
        raise ValueError(
            f"Only month(s) {sorted(covered)} were sampled. A month with no ladder is served "
            "uncorrected and nothing downstream says so; fetch the rest before shipping this."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    for column in columns:
        part = table[table["column"] == column]
        logger.info(
            "%s: served mean %.3f against cached %.3f over the ladder",
            column,
            part["served"].mean(),
            part["cached"].mean(),
        )
    logger.info("Wrote %s (%d row(s))", out, len(table))
    return table


def bias_fingerprint(config: Config) -> str | None:
    """Short digest of the correction in force, or `None` where there is none.

    A remotely sourced map scored before the quantile map existed, or against an older one,
    carries wind the current service would not produce. Nothing else in the sidecar would
    catch that: the model and its calibrator are untouched by refitting the correction.
    """
    import hashlib

    path = config.path(config.paths.bias_map_out)
    if not path.is_file():
        return None
    info = path.stat()
    payload = f"{int(info.st_mtime)}:{info.st_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def load_bias_map(config: Config) -> pd.DataFrame | None:
    path = config.path(config.paths.bias_map_out)
    if not path.is_file():
        logger.warning(
            "%s is missing, serving the remote fields uncorrected. Run `tfire fit-bias-map`.",
            path,
        )
        return None
    return pd.read_parquet(path)


def correct(frame: pd.DataFrame, table: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Map every corrected column of a remotely sourced daily frame onto the backbone ladder.

    Each value is read into its own cell-month served distribution and out of the cached one
    for the same cell and month. Outside the fitted range the ends are held flat, so an
    unprecedented value stays extreme instead of wrapping.
    """
    corrected = frame.copy()
    months = pd.DatetimeIndex(corrected["date"]).month.to_numpy()

    for column in config.bias.columns:
        if column not in corrected.columns:
            continue
        values = corrected[column].to_numpy(dtype="float64")
        mapped = values.copy()

        block = table[table["column"] == column]
        for (era5_id, month), rungs in block.groupby(["era5_id", "month"]):
            picked = (corrected["era5_id"].to_numpy() == era5_id) & (months == month)
            if not picked.any():
                continue
            ordered = rungs.sort_values("rung")
            served = ordered["served"].to_numpy(dtype="float64")
            cached = ordered["cached"].to_numpy(dtype="float64")
            # np.interp needs a strictly increasing x, and a flat stretch of the served
            # distribution (a month with no wind at all) is not
            usable = np.concatenate([[True], np.diff(served) > 0])
            mapped[picked] = np.interp(values[picked], served[usable], cached[usable])

        corrected[column] = mapped

    _keep_daily_ordering(corrected)
    return corrected


def _keep_daily_ordering(frame: pd.DataFrame) -> None:
    """A daily maximum mapped on its own ladder can land under the mean it belongs to."""
    if {"wind_speed_mean", "wind_speed_max"} <= set(frame.columns):
        frame["wind_speed_max"] = np.maximum(frame["wind_speed_max"], frame["wind_speed_mean"])


def bias_map_path(config: Config) -> Path:
    return config.path(config.paths.bias_map_out)
