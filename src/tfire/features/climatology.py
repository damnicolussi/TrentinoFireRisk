"""Per-cell monthly normals, so a day can be read as unusual for its own cell rather than high."""

from __future__ import annotations

import logging
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config

logger = logging.getLogger(__name__)

# the anomaly of each source column, named after it
ANOMALY_SUFFIX: Final = "_anom"

# rungs of the per-cell, per-month ladder a value is read against
RUNGS: Final = 101


def anomaly_column(source: str) -> str:
    return f"{source}{ANOMALY_SUFFIX}"


def _ladder() -> npt.NDArray[np.float64]:
    return np.linspace(0.0, 100.0, RUNGS)


def extract_climatology(config: Config, force: bool = False) -> pd.DataFrame:
    """The distribution of every anomaly source, per backbone cell and calendar month.

    Fitted on the training years alone. A reference period that reached into the holdout would
    let a test day help decide what counts as unusual for its own cell, which is leakage that
    no metric would show.
    """
    out = config.path(config.paths.climatology_out)
    if out.is_file() and not force:
        logger.info("%s already exists, use --force to rebuild", out)
        return pd.read_parquet(out)

    sources = list(config.climatology.columns)
    frame = _reference_frame(config, sources)

    ladder = _ladder()
    blocks = []
    for column in sources:
        for keys, part in frame.groupby([frame["era5_id"], frame["date"].dt.month]):
            era5_id, month = cast(tuple[int, int], keys)
            values = part[column].to_numpy(dtype="float64")
            usable = values[~np.isnan(values)]
            if usable.size == 0:
                continue
            blocks.append(
                pd.DataFrame(
                    {
                        "era5_id": np.int32(era5_id),
                        "month": np.int8(month),
                        "column": column,
                        "rung": ladder,
                        "value": np.percentile(usable, ladder),
                    }
                )
            )

    table = pd.concat(blocks, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    logger.info(
        "Wrote %s: %d column(s) over %d cell-month(s)",
        out,
        len(sources),
        table.groupby(["era5_id", "month"]).ngroups,
    )
    return table


def _reference_frame(config: Config, sources: list[str]) -> pd.DataFrame:
    """The training years of whichever cached tables carry the anomaly sources."""
    meteo = pd.read_parquet(config.path(config.paths.meteo_out))
    fwi = pd.read_parquet(config.path(config.paths.fwi_out))
    frame = meteo.merge(fwi, on=["era5_id", "date"], how="left", validate="1:1")

    missing = sorted(set(sources) - set(frame.columns))
    if missing:
        raise ValueError(f"No cached column for {missing}; check climatology.columns")

    first = pd.Timestamp(config.date_range.start)
    last = pd.Timestamp(f"{config.trentino.test_years_start - 1}-12-31")
    reference = frame[frame["date"].between(first, last)]
    logger.info("Normals from %s to %s: %d cell-day(s)", first.date(), last.date(), len(reference))
    return reference


def attach_anomalies(frame: pd.DataFrame, table: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add one anomaly column per source: where today sits in its own cell's month.

    Keyed on `era5_id`, the same backbone cell the FWI and the raw meteo already come from, so
    the anomaly and the value it is an anomaly of are read off the same series.
    """
    out = frame.copy()
    months = pd.DatetimeIndex(out["date"]).month.to_numpy()
    ladder = _ladder()

    for column in config.climatology.columns:
        target = anomaly_column(column)
        if column not in out.columns:
            raise ValueError(f"{column} is not on the frame, so {target} cannot be derived")

        values = out[column].to_numpy(dtype="float64")
        placed = np.full(values.shape, np.nan)

        block = table[table["column"] == column]
        for (era5_id, month), rungs in block.groupby(["era5_id", "month"]):
            picked = (out["era5_id"].to_numpy() == era5_id) & (months == month)
            if not picked.any():
                continue
            reference = rungs.sort_values("rung")["value"].to_numpy(dtype="float64")
            # a month whose normals are flat (no rain at all, ever) has no percentile to give
            usable = np.concatenate([[True], np.diff(reference) > 0])
            if usable.sum() < 2:
                placed[picked] = 50.0
                continue
            placed[picked] = np.interp(values[picked], reference[usable], ladder[usable])

        out[target] = placed.astype("float32")

    return out
