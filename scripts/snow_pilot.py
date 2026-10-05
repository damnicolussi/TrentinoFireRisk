"""ERA5-Land snow cover and depth on four half-years, checked before any full download.

Fetched into a directory of their own, so the backbone cache is untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests

from tfire.config import Config, load_config, setup_logging
from tfire.features.vegetation import operational_window
from tfire.report import table
from tfire.sources.era5land import Lattice, fetch_era5, half_months, open_half, read_lattice

logger = logging.getLogger("snow_pilot")

SNOW: Final = ["snow_cover", "snow_depth"]
PILOT_DIR: Final = Path("data/raw/era5land_snow_pilot")
HALVES: Final = {
    (1983, 1): "opening of the record",
    (2014, 1): "very snowy winter",
    (2022, 1): "nearly snowless winter",
    (2026, 1): "seam with Open-Meteo",
}
# the halves the Landsat check reads: both winters, inside the composites' record
LANDSAT_HALVES: Final = ((2014, 1), (2022, 1))

# `valid_fraction` counts clear pixels, and with vegetation.mask_snow on, snow is not clear: a
# filter on it alone keeps the snow-free cells. A cell counts where the month's scenes saw snow
# on it, or saw most of it clear; one never seen reads as zero snow and is left out.
MIN_CLEAR_FRACTION: Final = 0.5
SNOWY: Final = 0.5

# Open-Meteo: one in two weighted backbone cells, in two requests, well inside the free quota
OPEN_METEO_STRIDE: Final = 2
OPEN_METEO_BATCH: Final = 40
OPEN_METEO_PACE_S: Final = 65
SEAM: Final = (date(2026, 1, 1), date(2026, 6, 30))
# depths under a centimeter are noise on both sides
DEPTH_FLOOR_M: Final = 0.01

ELEVATION_BANDS: Final = (0, 1000, 1500, 2000, 4000)


def pilot_config(config: Config) -> Config:
    meteo = config.meteo.model_copy(update={"variables": SNOW})
    paths = config.paths.model_copy(update={"era5_raw": PILOT_DIR})
    return config.model_copy(update={"meteo": meteo, "paths": paths})


def daily_snow(config: Config, year: int, half: int) -> pd.DataFrame:
    """Local-day means per backbone cell: cover in percent, depth in meters."""
    dataset = open_half(config, year, half)
    days = (
        pd.DatetimeIndex(dataset["time"].to_numpy())
        + pd.Timedelta(hours=config.meteo.utc_offset_hours)
    ).normalize()
    n_cells = dataset.sizes["era5_id"]
    columns: dict[str, Any] = {}
    for short, name in (("snowc", "snow_cover"), ("sde", "snow_depth")):
        means = pd.DataFrame(dataset[short].to_numpy()).groupby(days).mean()
        columns[name] = means.to_numpy().reshape(-1)
    unique = days.unique()
    return pd.DataFrame(
        {
            "date": np.repeat(unique, n_cells),
            "era5_id": np.tile(np.arange(n_cells), len(unique)),
            **columns,
        }
    )


def coverage(config: Config, year: int, half: int, weighted: npt.NDArray[np.int64]) -> list[Any]:
    """Question 1: every hour present, no gaps over the cells the province leans on."""
    dataset = open_half(config, year, half)
    hours = dataset.sizes["time"]
    cover = dataset["snowc"].to_numpy()[:, weighted]
    depth = dataset["sde"].to_numpy()[:, weighted]
    missing = float(np.isnan(cover).mean() + np.isnan(depth).mean()) / 2
    february = pd.DatetimeIndex(dataset["time"].to_numpy()).month == 2
    return [
        f"{year} h{half}",
        HALVES[(year, half)],
        hours,
        f"{100 * missing:.2f}%",
        f"{np.nanmin(cover):.0f}-{np.nanmax(cover):.0f}",
        f"{np.nanmax(depth):.2f}",
        f"{np.nanmean(cover[february]):.1f}",
    ]


def interpolate(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Bilinear onto the grid cells, the way the backbone's other fields reach them."""
    pairs = weights.merge(daily, on="era5_id")
    for name in SNOW:
        pairs[name] = pairs[name] * pairs["weight"]
    return pairs.groupby(["cell_id", "date"], as_index=False)[SNOW].sum()


def landsat_check(
    config: Config,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    elevation: pd.Series,
    year: int,
    half: int,
) -> pd.DataFrame:
    """Question 2: monthly ERA5-Land cover at the cell against the month's Landsat snow fraction."""
    cells = interpolate(daily, weights)
    cells["month"] = cells["date"].dt.to_period("M")
    # the local-time shift carries the half's last hour into the next month
    cells = cells[
        cells["month"].dt.month.isin(half_months(half)) & (cells["month"].dt.year == year)
    ]
    monthly = cells.groupby(["cell_id", "month"], as_index=False)["snow_cover"].mean()

    rows = []
    for month in sorted(monthly["month"].unique()):
        landsat = operational_window(config, month.to_timestamp().date())
        usable = landsat[
            (landsat["snow_fraction"] > 0) | (landsat["valid_fraction"] >= MIN_CLEAR_FRACTION)
        ]
        joined = usable[["cell_id", "snow_fraction"]].merge(
            monthly[monthly["month"] == month], on="cell_id"
        )
        rows.append(joined)
    frame = pd.concat(rows, ignore_index=True)
    frame["elevation"] = elevation.reindex(frame["cell_id"]).to_numpy()
    return frame


def agreement_rows(frame: pd.DataFrame) -> list[list[Any]]:
    rows = []
    bands = pd.cut(frame["elevation"], ELEVATION_BANDS, right=False)
    for band, part in [("all", frame), *frame.groupby(bands, observed=True)]:
        landsat = part["snow_fraction"] >= SNOWY
        era5 = part["snow_cover"] / 100 >= SNOWY
        rows.append(
            [
                str(band),
                len(part),
                f"{part['snow_fraction'].mean():.2f}",
                f"{(part['snow_cover'] / 100).mean():.2f}",
                f"{part['snow_fraction'].corr(part['snow_cover'], method='spearman'):.2f}",
                f"{100 * float((landsat == era5).mean()):.1f}%",
                f"{100 * float((landsat & ~era5).mean()):.1f}%",
                f"{100 * float((~landsat & era5).mean()):.1f}%",
            ]
        )
    return rows


def open_meteo_depth(config: Config, lattice: Lattice, ids: npt.NDArray[np.int64]) -> pd.DataFrame:
    """Question 3: what the served path's archive model reports for snow depth over the seam."""
    cache = config.path(PILOT_DIR) / "open_meteo_snow_depth.json"
    if cache.is_file():
        payload: list[dict[str, Any]] = json.loads(cache.read_text(encoding="utf-8"))
    else:
        payload = []
        latitudes, longitudes = lattice.cell_latitudes(), lattice.cell_longitudes()
        for index, first in enumerate(range(0, len(ids), OPEN_METEO_BATCH)):
            if index:
                time.sleep(OPEN_METEO_PACE_S)
            batch = ids[first : first + OPEN_METEO_BATCH]
            params = {
                "latitude": ",".join(f"{value:.2f}" for value in latitudes[batch]),
                "longitude": ",".join(f"{value:.2f}" for value in longitudes[batch]),
                "start_date": SEAM[0].isoformat(),
                "end_date": SEAM[1].isoformat(),
                "hourly": "snow_depth",
                "models": config.forecast.archive_model,
                "timezone": "GMT",
            }
            response = requests.get(
                config.forecast.archive_url,
                params=params,
                timeout=config.forecast.request_timeout_s,
            )
            response.raise_for_status()
            answer = response.json()
            payload.extend(answer if isinstance(answer, list) else [answer])
            logger.info("Open-Meteo: %d of %d point(s)", len(payload), len(ids))
        cache.write_text(json.dumps(payload), encoding="utf-8")

    frames = []
    for era5_id, location in zip(ids, payload, strict=True):
        hourly = location["hourly"]
        stamps = pd.DatetimeIndex(hourly["time"]) + pd.Timedelta(
            hours=config.meteo.utc_offset_hours
        )
        series = pd.Series(hourly["snow_depth"], index=stamps, dtype="float64")
        day = series.groupby(stamps.normalize()).mean()
        frames.append(pd.DataFrame({"date": day.index, "era5_id": era5_id, "served": day.values}))
    return pd.concat(frames, ignore_index=True)


def seam_rows(joined: pd.DataFrame) -> tuple[list[list[Any]], dict[str, float]]:
    snowy = joined[(joined["snow_depth"] >= DEPTH_FLOOR_M) | (joined["served"] >= DEPTH_FLOOR_M)]
    ratio = float(snowy["served"].sum() / snowy["snow_depth"].sum())
    per_cell = snowy.groupby("era5_id")[["snow_depth", "served"]].mean()
    stats = {
        "points": float(joined["era5_id"].nunique()),
        "days": float(joined["date"].nunique()),
        "ratio": ratio,
        "pearson": float(snowy["served"].corr(snowy["snow_depth"])),
        "cell_pearson": float(per_cell["served"].corr(per_cell["snow_depth"])),
        "missing": float(joined["served"].isna().mean()),
    }
    rows = []
    for quantile in (0.5, 0.75, 0.9, 0.99):
        rows.append(
            [
                f"p{int(100 * quantile)}",
                f"{snowy['snow_depth'].quantile(quantile):.3f}",
                f"{snowy['served'].quantile(quantile):.3f}",
            ]
        )
    return rows, stats


def render(
    coverage_rows: list[list[Any]],
    agreement: list[list[Any]],
    quantiles: list[list[Any]],
    seam: dict[str, float],
    model: str,
) -> str:
    lines = [
        "# Snow on the backbone: pilot",
        "",
        "Generated by `scripts/snow_pilot.py`. ERA5-Land `snow_cover` (percent of "
        "the cell) and `snow_depth` (meters) from Earth Engine, the same collection and lattice "
        f"as the backbone, cached apart in `{PILOT_DIR}`. Nothing in the backbone changed.",
        "",
        "## 1. Are the bands there for the whole period?",
        "",
        "Every hour of the four half-years, over the backbone cells the province's weights use. "
        "The February column is the mean cover over those cells, the sanity check that the "
        "snowy and the snowless winter come out as such.",
        "",
    ]
    lines += table(
        [
            "half",
            "why",
            "hours",
            "missing",
            "cover range %",
            "max depth m",
            "February cover %",
        ],
        coverage_rows,
    )
    lines += [
        "",
        "## 2. Does the extraction hold at the cell?",
        "",
        "ERA5-Land cover, bilinear onto the 500 m cells and averaged over the month, against the "
        "Landsat snow fraction of the same month, 2014 h1 and 2022 h1. A cell-month counts where "
        f"the scenes saw snow on it or saw at least {MIN_CLEAR_FRACTION:.0%} of it clear; with "
        "snow masked out of the clear pixels, a filter on clear pixels alone would keep only "
        "the snow-free cells. Landsat sees the ground on clear days and ERA5-Land averages every "
        f"day, so the two are not the same quantity. Snowy means at least {SNOWY:.0%} on either "
        "side.",
        "",
    ]
    lines += table(
        [
            "elevation m",
            "cell-months",
            "Landsat mean",
            "ERA5-Land mean",
            "Spearman",
            "agree",
            "Landsat only",
            "ERA5-Land only",
        ],
        agreement,
    )
    lines += [
        "",
        "## 3. Does the served snow depth need a correction?",
        "",
        f"Open-Meteo archive, model `{model}`, the one the served path uses, against ERA5-Land "
        f"over {SEAM[0]} to {SEAM[1]}: {seam['points']:.0f} backbone points, "
        f"{seam['days']:.0f} days, daily means, days where either side has at least "
        f"{DEPTH_FLOOR_M * 100:.0f} cm. Served over ERA5-Land, summed: {seam['ratio']:.2f}. "
        f"Pearson on the days {seam['pearson']:.2f}, on the per-point means "
        f"{seam['cell_pearson']:.2f}. Missing served values: {100 * seam['missing']:.1f}%.",
        "",
    ]
    lines += table(["quantile", "ERA5-Land m", "served m"], quantiles)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)
    pilot = pilot_config(config)

    fetch_era5(pilot, sorted({year for year, _ in HALVES}), halves=list(HALVES))

    weights = pd.read_parquet(config.path(config.paths.era5_weights_out))
    weighted = np.sort(weights["era5_id"].unique()).astype("int64")
    elevation = pd.read_parquet(config.path(config.paths.topography_out)).set_index("cell_id")[
        "elevation_mean"
    ]

    coverage_rows = [coverage(pilot, year, half, weighted) for year, half in HALVES]

    checks = [
        landsat_check(config, daily_snow(pilot, year, half), weights, elevation, year, half)
        for year, half in LANDSAT_HALVES
    ]
    agreement = agreement_rows(pd.concat(checks, ignore_index=True))

    lattice = read_lattice(config)
    served = open_meteo_depth(config, lattice, weighted[::OPEN_METEO_STRIDE])
    era5 = daily_snow(pilot, *max(HALVES))
    joined = served.merge(era5, on=["date", "era5_id"], how="inner")
    quantiles, seam = seam_rows(joined)

    out = config.path(config.paths.report_dir) / "snow_pilot.md"
    out.write_text(
        render(coverage_rows, agreement, quantiles, seam, config.forecast.archive_model),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
