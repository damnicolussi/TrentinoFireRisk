"""Why the map runs high along some main roads: observed against expected, decade by decade.

Writes `reports/roads.md`. Road geometry comes from the OSM cache the human features were built
from, so nothing is downloaded. Expected ignitions come from the fit on the training years
alone, scored on one day in seven of the holdout years, and are normalized to the ignitions
observed there: the question is where the model puts its fires, not how many it predicts.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely.geometry.base import BaseGeometry

from tfire.config import Config, load_config, setup_logging
from tfire.features.human import calendar_features
from tfire.features.registry import load_registry
from tfire.grid import load_grid
from tfire.models.explain import registry_name
from tfire.models.trentino import align_columns, design_matrix
from tfire.report import table

logger = logging.getLogger("roads")

REPORT_FILENAME = "roads.md"

ROADS = ("SS349", "SS612", "SS42")
MAJOR_CLASSES = ("motorway", "trunk", "primary", "secondary")

# a corridor is the cells within this of the road; its flanks are cells 1-3 km away, at least
# 1 km from any major road and within 150 m of elevation, so the comparison is road against
# hillside rather than valley floor against mountain
CORRIDOR_M = 1000.0
NEAR_M = 350.0
FLANK_M = (1000.0, 3000.0)
FLANK_ELEVATION_M = 150.0
ELEVATION_BIN_M = 200.0

PERIODS = ((1984, 1993), (1994, 2003), (2004, 2014), (2015, 2024))
SHAP_DAYS = (date(2024, 3, 20), date(2024, 8, 15))
EXPECTED_STRIDE_DAYS = 7
LIGHTNING_CODE = 10


def major_roads(config: Config) -> gpd.GeoDataFrame:
    from tfire.sources.osm import LINE_TAGS, _bbox, _configure_cache, _features_from_bbox

    _configure_cache(config)
    raw = _features_from_bbox(_bbox(config), LINE_TAGS["roads"])
    kept = raw[raw["highway"].str.removesuffix("_link").isin(MAJOR_CLASSES)]
    lines = kept[kept.geometry.geom_type.isin(("LineString", "MultiLineString"))]
    return gpd.GeoDataFrame(lines[["highway", "ref"]], geometry=lines.geometry, crs=raw.crs).to_crs(
        config.crs
    )


def road_geometry(roads: gpd.GeoDataFrame, ref: str) -> BaseGeometry:
    # a way can carry two refs where routes share a carriageway, "SS349;SS350"
    tokens = roads["ref"].fillna("").str.split(";")
    return roads[tokens.apply(lambda refs: ref in refs)].geometry.union_all()


def ignitions(config: Config) -> pd.DataFrame:
    samples = pd.read_parquet(config.path(config.paths.samples_out))
    fires = pd.read_parquet(config.path(config.paths.fires_out), columns=["fire_id", "cause"])
    events = samples.loc[samples["is_fire"], ["cell_id", "date", "fire_id"]]
    events = events.merge(fires, on="fire_id", how="left", validate="m:1")
    calendar = calendar_features(pd.DatetimeIndex(events["date"].unique()))
    events = events.merge(calendar[["date", "season"]], on="date", how="left")
    events["year"] = events["date"].dt.year
    events["lightning"] = events["cause"] == LIGHTNING_CODE
    return events


def expected_mass(config: Config, cells: pd.Index) -> npt.NDArray[np.float64]:
    """Summed probability per cell from the training-years fit, over sampled holdout days."""
    from tfire.inference import GridScorer

    first = date(config.trentino.test_years_start, 1, 1)
    span = (config.date_range.end - first).days + 1
    days = [first + timedelta(days=k) for k in range(0, span, EXPECTED_STRIDE_DAYS)]
    scorer = GridScorer(config, days, holdout=True)
    mass = pd.Series(0.0, index=cells)
    for index, day in enumerate(days):
        scored = scorer.day(day)
        mass = mass.add(pd.Series(scored.probability, index=scored.frame["cell_id"]), fill_value=0)
        if index and index % 100 == 0:
            logger.info("  expected: %d/%d days", index, len(days))
    return np.asarray(mass.reindex(cells).fillna(0.0), dtype="float64")


def poisson_p(observed: int, expected: float) -> float:
    """Two-sided exact Poisson p-value, as twice the smaller tail."""
    from scipy.stats import poisson

    low = poisson.cdf(observed, expected)
    high = poisson.sf(observed - 1, expected)
    return float(min(1.0, 2 * min(low, high)))


def observed_against_expected(
    distance: npt.NDArray[np.float64],
    positions: npt.NDArray[np.int64],
    years: npt.NDArray[np.int64],
    mass: npt.NDArray[np.float64],
    first_test_year: int,
) -> dict[str, Any]:
    recent = years >= first_test_year
    inside = distance[positions] < CORRIDOR_M
    observed = int((inside & recent).sum())
    expected = float(mass[distance < CORRIDOR_M].sum() / mass.sum() * recent.sum())
    record = float(inside.sum() / len(positions) / ((distance < CORRIDOR_M).mean()))
    return {
        "expected": expected,
        "observed": observed,
        "p": poisson_p(observed, expected),
        "record_density": record,
    }


def by_period(
    corridor: npt.NDArray[np.bool_],
    elevation: npt.NDArray[np.float64],
    events: pd.DataFrame,
    positions: npt.NDArray[np.int64],
) -> list[dict[str, Any]]:
    """Corridor ignitions per period against the rest of the province at the same elevation.

    The expectation gives each corridor cell the per-cell rate of the non-corridor cells in its
    elevation band, period by period, so a ratio that falls over time is a decline the corridor
    has and the province at that height does not.
    """
    bins = np.floor(elevation / ELEVATION_BIN_M).astype("int64")
    rows = []
    for first, last in PERIODS:
        within = ((events["year"] >= first) & (events["year"] <= last)).to_numpy()
        at = positions[within]
        hit = np.bincount(bins[at][~corridor[at]], minlength=bins.max() + 1)
        cells = np.bincount(bins[~corridor], minlength=bins.max() + 1)
        rate = np.divide(hit, cells, out=np.zeros(len(hit)), where=cells > 0)
        expected = float(rate[bins[corridor]].sum())
        observed = int(corridor[at].sum())
        rows.append(
            {
                "period": f"{first}-{last}",
                "observed": observed,
                "expected": expected,
                "ratio": observed / expected if expected else math.nan,
            }
        )
    return rows


def composition(
    corridor: npt.NDArray[np.bool_], events: pd.DataFrame, positions: npt.NDArray[np.int64]
) -> list[dict[str, Any]]:
    """Season and lightning shares of the corridor's ignitions, beside the province's."""
    rows = []
    inside = corridor[positions]
    for first, last in PERIODS:
        within = ((events["year"] >= first) & (events["year"] <= last)).to_numpy()
        part, province = events[within & inside], events[within]
        rows.append(
            {
                "period": f"{first}-{last}",
                "n": len(part),
                **{
                    f"{season}": (part["season"] == season).mean() if len(part) else math.nan
                    for season in ("spring", "summer", "autumn", "winter")
                },
                "province_spring": (province["season"] == "spring").mean(),
                "province_winter": (province["season"] == "winter").mean(),
                "lightning": part["lightning"].mean() if len(part) else math.nan,
                "province_lightning": province["lightning"].mean(),
            }
        )
    return rows


def flanks(
    distance: npt.NDArray[np.float64],
    any_major: npt.NDArray[np.float64],
    xy: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64],
) -> tuple[list[int], npt.NDArray[np.int64]]:
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    near, flank = [], set()
    for index in np.flatnonzero(distance < NEAR_M):
        candidates = np.asarray(tree.query_ball_point(xy[index], FLANK_M[1]), dtype="int64")
        keep = candidates[
            (distance[candidates] > FLANK_M[0])
            & (any_major[candidates] > FLANK_M[0])
            & (np.abs(elevation[candidates] - elevation[index]) < FLANK_ELEVATION_M)
        ]
        if keep.size:
            near.append(int(index))
            flank.update(keep.tolist())
    return near, np.array(sorted(flank), dtype="int64")


def shap_groups(config: Config, day: date, cells: pd.Index) -> pd.DataFrame:
    """Per-cell SHAP contributions on one day, summed by feature category."""
    import xgboost as xgb

    from tfire.inference import GridScorer

    registry = load_registry()
    scorer = GridScorer(config, [day])
    scored = scorer.day(day)
    features, _, _ = design_matrix(scored.frame.assign(is_fire=False), registry)
    aligned = align_columns(features, scorer.columns)
    booster = scorer.estimator.get_booster()  # type: ignore[attr-defined]
    contributions = booster.predict(xgb.DMatrix(aligned), pred_contribs=True)[:, :-1]

    category = {spec.name: spec.category for spec in registry.features}
    known = set(category)
    groups = [category.get(registry_name(name, known), "other") for name in aligned.columns]
    frame = pd.DataFrame(contributions, columns=groups, index=scored.frame["cell_id"].to_numpy())
    return frame.T.groupby(level=0).sum().T.reindex(cells)


def render(sections: dict[str, Any], config: Config) -> str:
    test_first = config.trentino.test_years_start
    lines = [
        "# Main roads: SS349, SS612 and SS42",
        "",
        f"Expected ignitions within {CORRIDOR_M / 1000:g} km of each road come from the fit on "
        f"{config.date_range.start.year}-{test_first - 1} alone, scored on one day in "
        f"{EXPECTED_STRIDE_DAYS} of {test_first}-{config.date_range.end.year} and normalized to "
        "the ignitions observed there, so they say where the model puts its fires rather than "
        "how many. Record density is the corridor's share of all 1984-2024 ignitions over its "
        "share of cells.",
        "",
        *table(
            ("road", "expected", "observed", "p", "record density on average"),
            [
                [
                    road,
                    f"{block['oe']['expected']:.1f}",
                    block["oe"]["observed"],
                    f"{block['oe']['p']:.3f}",
                    f"{block['oe']['record_density']:.1f}",
                ]
                for road, block in sections.items()
            ],
        ),
        "",
        "## Ignitions per period against the province at the same elevation",
        "",
        "Each corridor cell is given the per-cell rate of the cells outside the corridor in its "
        f"{ELEVATION_BIN_M:g} m elevation band, period by period. A ratio that falls over time "
        "is a decline the corridor has and the province at that height does not.",
        "",
        *table(
            ("road", *[f"{a}-{b}" for a, b in PERIODS]),
            [
                [
                    road,
                    *[
                        f"{row['observed']} / {row['expected']:.1f} ({row['ratio']:.2f})"
                        for row in block["periods"]
                    ],
                ]
                for road, block in sections.items()
            ],
        ),
        "",
        "Cells read observed / expected (ratio).",
        "",
        "## Season and cause",
        "",
    ]
    for road, block in sections.items():
        lines += [
            f"**{road}**",
            "",
            *table(
                (
                    "period",
                    "n",
                    "spring",
                    "summer",
                    "autumn",
                    "winter",
                    "province spring",
                    "province winter",
                    "lightning",
                    "province lightning",
                ),
                [
                    [
                        row["period"],
                        row["n"],
                        *[
                            "" if math.isnan(row[key]) else f"{row[key]:.0%}"
                            for key in ("spring", "summer", "autumn", "winter")
                        ],
                        f"{row['province_spring']:.0%}",
                        f"{row['province_winter']:.0%}",
                        "" if math.isnan(row["lightning"]) else f"{row['lightning']:.0%}",
                        f"{row['province_lightning']:.0%}",
                    ]
                    for row in block["composition"]
                ],
            ),
            "",
        ]
    lines += [
        "## What the model reads in the corridor",
        "",
        f"Mean SHAP contribution (log-odds) of cells within {NEAR_M:g} m of the road minus that "
        f"of flank cells {FLANK_M[0] / 1000:g}-{FLANK_M[1] / 1000:g} km away, at least "
        f"{FLANK_M[0] / 1000:g} km from any major road and within {FLANK_ELEVATION_M:g} m of "
        "elevation, by feature category. The shipped estimator, on the backbone weather.",
        "",
    ]
    groups = sorted({name for block in sections.values() for name in block["shap"][0].index})
    for index, day in enumerate(SHAP_DAYS):
        lines += [
            f"**{day:%d/%m/%Y}**",
            "",
            *table(
                ("category", *sections),
                [
                    [
                        name,
                        *[
                            f"{block['shap'][index].get(name, 0.0):+.2f}"
                            for block in sections.values()
                        ],
                    ]
                    for name in groups
                ]
                + [
                    [
                        "total",
                        *[f"{block['shap'][index].sum():+.2f}" for block in sections.values()],
                    ]
                ],
            ),
            "",
        ]
    lines += [
        "Corridor and flank cells: "
        + ", ".join(
            f"{road} {block['near']} and {block['flank']}" for road, block in sections.items()
        )
        + ".",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)

    _, grid = load_grid(config)
    active = grid[grid["is_trentino"]].set_index("cell_id")
    cells = active.index
    xy = active[["x_coordinate", "y_coordinate"]].to_numpy("float64")
    topography = pd.read_parquet(config.path(config.paths.topography_out)).set_index("cell_id")
    elevation = topography["elevation_mean"].reindex(cells).to_numpy("float64")
    points = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=config.crs)

    roads = major_roads(config)
    any_major = points.distance(roads.geometry.union_all()).to_numpy()

    events = ignitions(config)
    positions = cells.get_indexer(pd.Index(events["cell_id"]))
    events, positions = events[positions >= 0], positions[positions >= 0]
    years = events["year"].to_numpy()

    logger.info("Scoring the holdout years for the expected ignitions")
    mass = expected_mass(config, cells)
    shap_by_day = [shap_groups(config, day, cells) for day in SHAP_DAYS]

    sections: dict[str, Any] = {}
    for road in ROADS:
        distance = points.distance(road_geometry(roads, road)).to_numpy()
        corridor = distance < CORRIDOR_M
        near, flank = flanks(distance, any_major, xy, elevation)
        sections[road] = {
            "oe": observed_against_expected(
                distance, positions, years, mass, config.trentino.test_years_start
            ),
            "periods": by_period(corridor, elevation, events, positions),
            "composition": composition(corridor, events, positions),
            "shap": [frame.iloc[near].mean() - frame.iloc[flank].mean() for frame in shap_by_day],
            "near": len(near),
            "flank": len(flank),
        }
        logger.info("%s: %d corridor cell(s)", road, int(corridor.sum()))

    out = config.path(config.paths.report_dir) / REPORT_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(sections, config), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
