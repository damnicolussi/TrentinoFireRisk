"""ERA5 convective available potential energy on five half-years, checked before any full download.

Fetched onto the backbone lattice into a directory of their own, so the backbone cache is untouched.
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
from sklearn.metrics import roc_auc_score

from tfire.config import Config, load_config, setup_logging
from tfire.report import table
from tfire.sources.era5land import Lattice, fetch_era5, open_half, read_lattice

logger = logging.getLogger("convection_pilot")

VARIABLES: Final = ["convective_available_potential_energy", "convective_precipitation"]
PILOT_DIR: Final = Path("data/raw/era5_convection_pilot")
HALVES: Final = {
    (1990, 2): "early record, 19 lightning fires",
    (2003, 2): "hot summer, 27 lightning fires",
    (2015, 2): "holdout, 12 lightning fires",
    (2017, 2): "holdout, 9 lightning fires",
    (2024, 2): "seam with the served forecast",
}
SIGNAL_HALVES: Final = ((1990, 2), (2003, 2), (2015, 2), (2017, 2))
LIGHTNING_CODE: Final = 10
SUMMER_MONTHS: Final = (7, 8, 9)

# lightning fires smolder before anyone reports them, so the days before count as well
LAG_DAYS: Final = 3

# the forecast endpoint's archive of IFS runs, the only Open-Meteo source that serves CAPE
HISTORICAL_FORECAST_URL: Final = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SEAM: Final = (date(2024, 7, 1), date(2024, 9, 30))
OPEN_METEO_STRIDE: Final = 2
OPEN_METEO_BATCH: Final = 40
OPEN_METEO_PACE_S: Final = 65

# below this a day carries no convection worth comparing between sources
ACTIVE_CAPE: Final = 50.0


def pilot_config(config: Config) -> Config:
    meteo = config.meteo.model_copy(update={"variables": VARIABLES})
    paths = config.paths.model_copy(update={"era5_raw": PILOT_DIR})
    return config.model_copy(update={"meteo": meteo, "paths": paths})


def daily_convection(config: Config, year: int, half: int) -> pd.DataFrame:
    """Per backbone cell and whole local day: CAPE maximum, convective rain, and the lag maximum."""
    dataset = open_half(config, year, half)
    stamps = pd.DatetimeIndex(dataset["time"].to_numpy()) + pd.Timedelta(
        hours=config.meteo.utc_offset_hours
    )
    days = stamps.normalize()
    whole = days.isin(days.value_counts()[lambda counts: counts == 24].index)
    days = days[whole]

    cape = pd.DataFrame(dataset["cape"].to_numpy()[whole]).groupby(days).max()
    # hourly amounts, with Earth Engine's rounding leaving a few just under zero
    rain = pd.DataFrame(np.maximum(dataset["cp"].to_numpy()[whole], 0.0) * 1000).groupby(days).sum()
    lag = cape.shift(1).rolling(LAG_DAYS, min_periods=LAG_DAYS).max()

    n_cells = cape.shape[1]
    return pd.DataFrame(
        {
            "date": np.repeat(cape.index, n_cells),
            "era5_id": np.tile(np.arange(n_cells), len(cape)),
            "cape_max": cape.to_numpy().reshape(-1),
            "cape_lag": lag.to_numpy().reshape(-1),
            "conv_rain": rain.to_numpy().reshape(-1),
        }
    )


COLUMNS: Final = ("cape_max", "cape_lag", "conv_rain")


def interpolate(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Bilinear onto the grid cells, the way the backbone's other fields reach them."""
    pairs = weights.merge(daily, on="era5_id")
    for name in COLUMNS:
        pairs[name] = pairs[name] * pairs["weight"]
    return pairs.groupby(["cell_id", "date"], as_index=False)[list(COLUMNS)].sum(min_count=1)


def coverage(config: Config, year: int, half: int, weighted: npt.NDArray[np.int64]) -> list[Any]:
    """Question 1: every hour present, no gaps over the cells the province leans on."""
    dataset = open_half(config, year, half)
    cape = dataset["cape"].to_numpy()[:, weighted]
    rain = dataset["cp"].to_numpy()[:, weighted]
    missing = float(np.isnan(cape).mean() + np.isnan(rain).mean()) / 2
    july = pd.DatetimeIndex(dataset["time"].to_numpy()).month == 7
    return [
        f"{year} h{half}",
        HALVES[(year, half)],
        dataset.sizes["time"],
        f"{100 * missing:.2f}%",
        f"{np.nanmax(cape):.0f}",
        f"{np.nanmean(np.nanmax(cape[july], axis=1)):.0f}",
        f"{1000 * np.nansum(np.maximum(rain, 0)) / rain.shape[1]:.0f}",
    ]


def labeled_samples(config: Config) -> pd.DataFrame:
    """Training-table cell-days with the fire cause attached; several fires on one row are one."""
    samples = pd.read_parquet(config.path(config.paths.samples_out))
    fires = pd.read_parquet(config.path(config.paths.fires_out), columns=["fire_id", "cause"])
    samples = samples.merge(fires, on="fire_id", how="left")
    samples["kind"] = np.where(
        ~samples["is_fire"],
        "negative",
        np.where(samples["cause"].eq(LIGHTNING_CODE).fillna(False), "lightning", "other fire"),
    )
    return samples.drop_duplicates(["cell_id", "date"])


def separation_rows(frame: pd.DataFrame) -> list[list[Any]]:
    """Question 2a: does a cell-day's own convection set lightning ignitions apart?"""
    negatives = frame[frame["kind"] == "negative"]
    rows = []
    for kind in ("lightning", "other fire"):
        positives = frame[frame["kind"] == kind]
        row: list[Any] = [kind, len(positives)]
        for name in COLUMNS:
            both = pd.concat([positives, negatives]).dropna(subset=[name])
            labels = (both["kind"] == kind).to_numpy()
            row.append(f"{roc_auc_score(labels, both[name]):.3f}")
            row.append(f"{positives[name].median():.0f}")
        rows.append(row)
    row = ["negative", len(negatives)]
    for name in COLUMNS:
        row += ["", f"{negatives[name].median():.0f}"]
    rows.append(row)
    return rows


def within_day(
    config: Config, daily: pd.DataFrame, weights: pd.DataFrame, ignitions: pd.DataFrame
) -> pd.DataFrame:
    """Question 2b: where each lightning ignition's cell ranks among all cells on its own day."""
    wanted = daily[daily["date"].isin(ignitions["date"].unique())]
    grid = interpolate(wanted, weights)
    out = []
    for day, cells in grid.groupby("date"):
        ranks = cells.set_index("cell_id")[list(COLUMNS)].rank(pct=True)
        spread = cells[["cape_max"]].describe(percentiles=[0.1, 0.9]).T
        hits = ignitions[ignitions["date"] == day]
        found = ranks.reindex(hits["cell_id"])
        found["date"] = day
        found["cell_id"] = hits["cell_id"].to_numpy()
        found["cape_p10"] = float(spread["10%"].iloc[0])
        found["cape_p90"] = float(spread["90%"].iloc[0])
        out.append(found.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def reference_percentiles(config: Config) -> pd.Series:
    """v2's within-day percentile of each holdout ignition, as the variants harness stored it."""
    path = config.path(config.paths.trentino_model_dir) / "experiments" / "recency" / "results.json"
    if not path.is_file():
        return pd.Series(dtype="float64")
    axis = json.loads(path.read_text(encoding="utf-8"))["axis"]
    events = axis["events"]
    index = pd.MultiIndex.from_arrays(
        [np.asarray(events["cell_id"]), pd.to_datetime(events["date"])], names=["cell_id", "date"]
    )
    series = pd.Series(axis["variants"]["v2"]["percentiles"], index=index, dtype="float64")
    return series[~series.index.duplicated()]


def served_cape(config: Config, lattice: Lattice, ids: npt.NDArray[np.int64]) -> pd.DataFrame:
    """Question 3: IFS CAPE as Open-Meteo archived it, daily maximum per backbone cell."""
    cache = config.path(PILOT_DIR) / "open_meteo_cape.json"
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
                "hourly": "cape",
                "models": config.forecast.forecast_model,
                "timezone": "GMT",
            }
            response = requests.get(
                HISTORICAL_FORECAST_URL, params=params, timeout=config.forecast.request_timeout_s
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
        series = pd.Series(hourly["cape"], index=stamps, dtype="float64")
        day = series.groupby(stamps.normalize()).max()
        frames.append(pd.DataFrame({"date": day.index, "era5_id": era5_id, "served": day.values}))
    return pd.concat(frames, ignore_index=True)


def seam_rows(joined: pd.DataFrame) -> tuple[list[list[Any]], dict[str, float]]:
    active = joined[(joined["cape_max"] >= ACTIVE_CAPE) | (joined["served"] >= ACTIVE_CAPE)]
    per_cell = joined.groupby("era5_id")[["cape_max", "served"]].mean()
    per_day = joined.groupby("date")[["cape_max", "served"]].mean()
    stats = {
        "points": float(joined["era5_id"].nunique()),
        "days": float(joined["date"].nunique()),
        "ratio": float(active["served"].sum() / active["cape_max"].sum()),
        "pearson": float(active["served"].corr(active["cape_max"])),
        "spearman": float(active["served"].corr(active["cape_max"], method="spearman")),
        "day_pearson": float(per_day["served"].corr(per_day["cape_max"])),
        "cell_pearson": float(per_cell["served"].corr(per_cell["cape_max"])),
        "missing": float(joined["served"].isna().mean()),
    }
    rows = [
        [
            f"p{int(100 * quantile)}",
            f"{joined['cape_max'].quantile(quantile):.0f}",
            f"{joined['served'].quantile(quantile):.0f}",
        ]
        for quantile in (0.5, 0.75, 0.9, 0.99)
    ]
    return rows, stats


def render(
    coverage_rows: list[list[Any]],
    separation: list[list[Any]],
    ranks: pd.DataFrame,
    quantiles: list[list[Any]],
    seam: dict[str, float],
    model: str,
    n_cells: int,
) -> str:
    lines = [
        "# Convection on the backbone: pilot",
        "",
        "Generated by `scripts/convection_pilot.py`. ERA5 `convective_available_potential_energy` "
        "(J/kg) and `convective_precipitation` (hourly, m) from Earth Engine's "
        "`ECMWF/ERA5/HOURLY`, resampled bilinearly from 0.25 degrees onto the backbone's 0.1 "
        f"degree lattice and cached apart in `{PILOT_DIR}`. Nothing in the backbone changed. "
        f"Per local day: `cape_max` the hourly maximum, `cape_lag` the maximum of the {LAG_DAYS} "
        "days before, `conv_rain` the convective rain in mm.",
        "",
        "## 1. Are the bands there for the whole period?",
        "",
        "Earth Engine holds every hour from 1983 to 2025 (8760 or 8784 a year) and 2026 to "
        "28 September. The fetched halves, over the backbone cells the province's weights use:",
        "",
    ]
    lines += table(
        [
            "half",
            "why",
            "hours",
            "missing",
            "CAPE max (J/kg)",
            "July mean daily max",
            "convective rain per cell (mm)",
        ],
        coverage_rows,
    )
    lines += [
        "",
        "## 2. Does convection set the lightning ignitions apart?",
        "",
        "### Against the training table's negatives",
        "",
        "Cell-days of the training table in July to September of the four signal halves. AUROC "
        "of each value alone, lightning ignitions and the other ignitions each against the same "
        "negatives, and the median value per group.",
        "",
    ]
    lines += table(
        [
            "group",
            "cell-days",
            "AUROC cape_max",
            "median cape_max",
            "AUROC cape_lag",
            "median cape_lag",
            "AUROC conv_rain",
            "median conv_rain",
        ],
        separation,
    )
    has_v2 = ranks["v2"].notna()
    lines += [
        "",
        "### Within the day",
        "",
        "The event axis the models are judged on ranks a cell against every other cell on the "
        "same day, so what counts is whether convection varies across the province, not only "
        f"between days. Percentile of each lightning ignition's cell among all {n_cells:,} grid "
        "cells on its own day, all months of the signal halves.",
        "",
    ]
    rank_rows = [
        [
            "median percentile",
            len(ranks),
            *(f"{ranks[name].median():.3f}" for name in COLUMNS),
            f"{ranks.loc[has_v2, 'v2'].median():.3f} ({int(has_v2.sum())})" if has_v2.any() else "",
        ],
        [
            "share at or above the 90th",
            len(ranks),
            *(f"{100 * float((ranks[name] >= 0.9).mean()):.0f}%" for name in COLUMNS),
            f"{100 * float((ranks.loc[has_v2, 'v2'] >= 0.9).mean()):.0f}%" if has_v2.any() else "",
        ],
    ]
    lines += table(
        ["", "ignitions", *COLUMNS, "v2, holdout ones only"],
        rank_rows,
    )
    lines += [
        "",
        f"On those days the province's `cape_max` runs from {ranks['cape_p10'].median():.0f} "
        f"J/kg at the 10th percentile of cells to {ranks['cape_p90'].median():.0f} at the 90th "
        "(medians over the days).",
        "",
        "## 3. Does the served CAPE sit on ERA5's?",
        "",
        "The archive endpoint the served path uses for the past does not carry CAPE, and "
        f"`showers` from `{model}` reads zero all summer, so only CAPE can be served, from the "
        f"forecast endpoint. Its archive of `{model}` runs, at {seam['points']:.0f} backbone "
        f"cells (one in {OPEN_METEO_STRIDE}) over {seam['days']:.0f} days from {SEAM[0]} to "
        f"{SEAM[1]}, against ERA5's daily maximum at the same cells. Missing: "
        f"{100 * seam['missing']:.1f}%.",
        "",
        f"- On cell-days where either source reaches {ACTIVE_CAPE:.0f} J/kg, served over ERA5 is "
        f"{seam['ratio']:.2f}, Pearson {seam['pearson']:.2f}, Spearman {seam['spearman']:.2f}.",
        f"- Province mean per day, Pearson {seam['day_pearson']:.2f}; mean per cell over the "
        f"period, Pearson {seam['cell_pearson']:.2f}.",
        "",
    ]
    lines += table(["quantile", "ERA5 cape_max", "served cape_max"], quantiles)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    config = load_config()
    setup_logging(config)
    pilot = pilot_config(config)

    fetch_era5(pilot, sorted({year for year, _ in HALVES}), halves=list(HALVES))

    weights = pd.read_parquet(config.path(config.paths.era5_weights_out))
    weighted = np.sort(weights["era5_id"].unique())
    coverage_rows = [coverage(pilot, year, half, weighted) for year, half in HALVES]

    daily = {key: daily_convection(pilot, *key) for key in HALVES}
    signal = pd.concat([daily[key] for key in SIGNAL_HALVES], ignore_index=True)

    samples = labeled_samples(config)
    samples = samples[
        samples["date"].dt.month.isin(SUMMER_MONTHS)
        & samples["date"].dt.year.isin({year for year, _ in SIGNAL_HALVES})
    ]
    cells = interpolate(signal, weights[weights["cell_id"].isin(samples["cell_id"])])
    frame = samples.merge(cells, on=["cell_id", "date"], how="inner")
    separation = separation_rows(frame)

    lightning = labeled_samples(config)
    lightning = lightning[
        (lightning["kind"] == "lightning") & lightning["date"].isin(signal["date"].unique())
    ][["cell_id", "date"]]
    ranks = within_day(config, signal, weights, lightning)
    reference = reference_percentiles(config)
    ranks["v2"] = reference.reindex(pd.MultiIndex.from_frame(ranks[["cell_id", "date"]])).to_numpy()

    lattice = read_lattice(config)
    ids = weighted[::OPEN_METEO_STRIDE]
    served = served_cape(config, lattice, ids)
    era5 = daily[max(HALVES)]
    joined = era5.merge(served, on=["era5_id", "date"], how="inner")
    quantiles, seam = seam_rows(joined)

    text = render(
        coverage_rows,
        separation,
        ranks,
        quantiles,
        seam,
        config.forecast.forecast_model,
        weights["cell_id"].nunique(),
    )
    out = config.path(config.paths.report_dir) / "convection_pilot.md"
    out.write_text(text, encoding="utf-8")
    ranks.to_csv(config.path(PILOT_DIR) / "lightning_ranks.csv", index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
