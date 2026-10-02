"""Runs of extreme cells after the record, and whether the fires they imply turned up."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import GridSpec, load_grid
from tfire.models.cases import Case, cadastre_cases, locate, read_cases
from tfire.models.danger import DangerClasses, load_danger_classes
from tfire.models.events import history_baseline

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

EPISODES_REPORT: Final = "episodes_2025_2026.md"
EPISODES_TABLE: Final = "episodes_2025_2026.csv"

LEVELS: Final = (99.0, 99.9)

_TOP_EPISODES: Final = 50

_CONNECTIVITY: Final = np.ones((3, 3, 3), dtype=bool)


def label_episodes(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int32], int]:
    """Connected components of a `(day, row, col)` mask."""
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=_CONNECTIVITY)
    return np.asarray(labels, dtype="int32"), int(count)


def raw_threshold(classes: DangerClasses, level: float) -> float:
    if classes.raw_quantiles is None:
        raise ValueError("danger_classes.json has no raw-score ladder. Run `tfire danger-classes`.")
    steps = len(classes.raw_quantiles) - 1
    return float(classes.raw_quantiles[int(round(level / 100 * steps))])


def garwood(count: int) -> tuple[float, float]:
    from scipy.stats import chi2

    low = chi2.ppf(0.025, 2 * count) / 2 if count else 0.0
    return float(low), float(chi2.ppf(0.975, 2 * count + 2) / 2)


def _table_path(config: Config, day: date) -> Path:
    return config.path(config.paths.risk_dir) / f"risk_{day:%Y-%m-%d}.parquet"


def _is_current(config: Config, day: date, fingerprint: str) -> bool:
    import pyarrow.parquet as pq

    table = _table_path(config, day)
    meta = table.with_suffix(".json")
    if not (table.is_file() and meta.is_file()):
        return False
    from tfire.inference import superseded_by_archive

    sidecar = json.loads(meta.read_text(encoding="utf-8"))
    if superseded_by_archive(config, day, list(sidecar.get("sources", [])), date.today()):
        return False
    return bool(
        sidecar.get("model_fingerprint") == fingerprint and "score" in pq.read_schema(table).names
    )


def ensure_maps(config: Config, days: list[date]) -> None:
    """Predict every day whose map is missing or was drawn by another model or rank rule."""
    from tfire.inference import cached_span, model_fingerprint, predict_days

    fingerprint = model_fingerprint(config)
    stale = [day for day in days if not _is_current(config, day, fingerprint)]
    if not stale:
        return
    _, cached_end = cached_span(config)

    runs: list[list[date]] = []
    for day in stale:
        if (
            runs
            and day - runs[-1][-1] == timedelta(days=1)
            and (day <= cached_end) == (runs[-1][-1] <= cached_end)
        ):
            runs[-1].append(day)
        else:
            runs.append([day])
    logger.info("Predicting %d day(s) in %d run(s)", len(stale), len(runs))
    for run in runs:
        predict_days(config, run[0], days=len(run), force=True)


def known_fires(config: Config, path: Path | None) -> pd.DataFrame:
    """Fires after the record, on their cells: the cadastre's, and the file's where given."""
    spec, _ = load_grid(config)
    cases: list[tuple[Case, str]] = [(case, "cadastre") for case in cadastre_cases(config)]
    if path:
        cases += [(case, "file") for case in read_cases(path)]

    rows = []
    for case, origin in cases:
        _, _, cell = locate(spec, config.crs, case.lat, case.lon)
        rows.append(
            {
                "id": case.id,
                "date": case.date,
                "cell_id": cell,
                "origin": origin,
                "year": case.date.year,
            }
        )
    return pd.DataFrame(rows, columns=["id", "date", "cell_id", "origin", "year"])


def _municipalities(config: Config) -> gpd.GeoDataFrame | None:
    relative = config.paths.pat_municipalities
    if relative is None or not config.path(relative).is_file():
        logger.warning("No municipality layer configured or on disk, episodes stay unnamed")
        return None
    import geopandas as gpd

    layer = gpd.read_file(config.path(relative)).to_crs(config.crs)
    names = [
        column
        for column in layer.columns
        if any(key in column.lower() for key in ("nome", "name", "comune", "desc"))
    ]
    if not names:
        logger.warning("The municipality layer has no name column, episodes stay unnamed")
        return None
    return layer[[names[0], "geometry"]].rename(columns={names[0]: "municipality"})


def _name_at(layer: gpd.GeoDataFrame | None, x: float, y: float) -> str | None:
    if layer is None:
        return None
    from shapely.geometry import Point

    hit = layer[layer.contains(Point(x, y))]
    return None if hit.empty else str(hit["municipality"].iloc[0])


def find_episodes(
    config: Config,
    days: list[date],
    classes: DangerClasses,
    spec: GridSpec,
    level: float,
) -> pd.DataFrame:
    """One row per extreme cell-day, labelled with the episode it belongs to."""
    threshold = raw_threshold(classes, level)
    mask = np.zeros((len(days), spec.n_rows, spec.n_cols), dtype=bool)
    probability = np.zeros(mask.shape, dtype="float32")

    for index, day in enumerate(days):
        table = pd.read_parquet(
            _table_path(config, day), columns=["cell_id", "probability", "score"]
        )
        hot = table[table["score"] >= threshold]
        col, row = spec.cell_index(hot["cell_id"].to_numpy())
        mask[index, row, col] = True
        probability[index, row, col] = hot["probability"].to_numpy()

    labels, count = label_episodes(mask)
    day_index, row, col = np.nonzero(labels)
    logger.info("Level %.1f: %d extreme cell-day(s) in %d episode(s)", level, day_index.size, count)
    return pd.DataFrame(
        {
            "episode": labels[day_index, row, col],
            "date": pd.to_datetime([days[i] for i in day_index]),
            "cell_id": row.astype("int64") * spec.n_cols + col,
            "probability": probability[day_index, row, col].astype("float64"),
        }
    )


def summarize_episodes(
    cells: pd.DataFrame,
    fires: pd.DataFrame,
    spec: GridSpec,
    config: Config,
    base_url: str,
) -> pd.DataFrame:
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(config.crs, "EPSG:4326", always_xy=True)
    layer = _municipalities(config)
    hits = cells.merge(
        fires.assign(date=pd.to_datetime(fires["date"])), on=["date", "cell_id"], how="inner"
    )

    rows = []
    for _, part in cells.groupby("episode"):
        episode = int(part["episode"].iloc[0])
        x, y = spec.cell_center(part["cell_id"].to_numpy())
        cx, cy = float(np.mean(x)), float(np.mean(y))
        lon, lat = to_wgs84.transform(cx, cy)
        first = part["date"].min()
        found = hits[hits["episode"] == episode]
        rows.append(
            {
                "episode": episode,
                "start": first.date(),
                "end": part["date"].max().date(),
                "cells": int(part["cell_id"].nunique()),
                "cell_days": len(part),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "municipality": _name_at(layer, cx, cy),
                "link": f"{base_url.rstrip('/')}/#map/{first:%Y-%m-%d}",
                "expected": float(part["probability"].sum()),
                "observed": len(found),
                "fires": ", ".join(found["id"]),
            }
        )
    return pd.DataFrame(rows)


def _baseline_hits(cells: pd.DataFrame, fires: pd.DataFrame, baseline: pd.Series) -> int:
    """Fires landing in the day's top fire-history cells, as many cells as the day flagged."""
    ordered = baseline.sort_values(ascending=False).index.to_numpy()
    per_day = cells.groupby("date").size()
    count = 0
    for day, cell in zip(pd.to_datetime(fires["date"]), fires["cell_id"], strict=True):
        flagged = int(per_day.get(day, 0))
        if flagged and cell in set(ordered[:flagged].tolist()):
            count += 1
    return count


def _calibration_rows(
    cells: pd.DataFrame, fires: pd.DataFrame, baseline: pd.Series
) -> list[dict[str, Any]]:
    rows = []
    for year, part in cells.groupby(cells["date"].dt.year):
        known = fires[fires["year"] == year]
        inside = part.merge(
            known.assign(date=pd.to_datetime(known["date"])), on=["date", "cell_id"], how="inner"
        )
        observed = len(inside)
        low, high = garwood(observed)
        rows.append(
            {
                "year": int(year),
                "sources": ", ".join(sorted(known["origin"].unique())) or "none",
                "episodes": int(part["episode"].nunique()),
                "cell_days": len(part),
                "expected": float(part["probability"].sum()),
                "observed": observed,
                "observed_low": low,
                "observed_high": high,
                "known_fires": len(known),
                "known_in_episode": observed,
                "known_in_history_top": _baseline_hits(part, known, baseline),
            }
        )
    return rows


def build_episodes(
    config: Config,
    start: date,
    end: date,
    path: Path | None,
    base_url: str,
) -> Path:
    """Extreme-cell episodes from `start` to `end`, checked against the fires that followed."""
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    ensure_maps(config, days)

    spec, _ = load_grid(config)
    classes = load_danger_classes(config)
    fires = known_fires(config, path)
    fires = fires[(fires["date"] >= start) & (fires["date"] <= end) & fires["cell_id"].notna()]
    baseline = history_baseline(config, config.date_range.end.year)

    sections = []
    top = pd.DataFrame()
    for level in LEVELS:
        cells = find_episodes(config, days, classes, spec, level)
        episodes = summarize_episodes(cells, fires, spec, config, base_url)
        sections.append((level, episodes, _calibration_rows(cells, fires, baseline)))
        if level == LEVELS[0]:
            top = episodes.sort_values("expected", ascending=False).head(_TOP_EPISODES)

    directory = config.path(config.paths.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    top.to_csv(directory / EPISODES_TABLE, index=False)
    report = directory / EPISODES_REPORT
    report.write_text(render_episodes(sections, start, end, config), encoding="utf-8")
    logger.info("Wrote %s and %s", report, directory / EPISODES_TABLE)
    return report


def render_episodes(
    sections: list[tuple[float, pd.DataFrame, list[dict[str, Any]]]],
    start: date,
    end: date,
    config: Config,
) -> str:
    from tfire.report import table

    lines = [
        "# Extreme-risk episodes after the record",
        "",
        f"Every day from {start:%Y-%m-%d} to {end:%Y-%m-%d}, scored by the shipped "
        f"{config.trentino.version} model. A cell-day is extreme when its score reaches the "
        "given percentile of the record; extreme cell-days that touch in space (8 neighbors) or "
        "on consecutive days form one episode. Expected fires are the sum of the calibrated "
        "probabilities over an episode's cell-days, so observed against expected is a check of "
        "calibration, not of whether a fire was foreseen.",
        "",
        "Years are kept apart: the cadastre covers 2025, while later fires come from press "
        "reports and official statements until the cadastre reaches them, which undercounts "
        "the observed column.",
        "",
    ]
    for level, episodes, rows in sections:
        lines += [
            f"## Above the {level:g}th percentile",
            "",
            *table(
                (
                    "year",
                    "fire sources",
                    "episodes",
                    "cell-days",
                    "expected",
                    "observed",
                    "95% interval",
                    "known fires",
                    "in an episode",
                    "in the history top",
                ),
                [
                    [
                        row["year"],
                        row["sources"],
                        row["episodes"],
                        row["cell_days"],
                        f"{row['expected']:.2f}",
                        row["observed"],
                        f"{row['observed_low']:.1f}-{row['observed_high']:.1f}",
                        row["known_fires"],
                        row["known_in_episode"],
                        row["known_in_history_top"],
                    ]
                    for row in rows
                ],
            ),
            "",
            f"{len(episodes)} episode(s). The history top is the same number of cells per day, "
            "taken from the top of the fire-history density instead.",
            "",
        ]
    return "\n".join(lines)
