"""Human activity: OSM infrastructure/POIs, Natura 2000, WorldPop density, and calendar
features computed with no external source.
"""

from __future__ import annotations

import logging
from typing import Final

import geopandas as gpd
import holidays
import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import STRtree, points
from shapely.geometry.base import BaseGeometry

from tfire.config import Config
from tfire.features.geography import waterbody_fraction
from tfire.geo import verify_crs
from tfire.grid import GridSpec, load_grid
from tfire.raster import read_cell_bands
from tfire.sources import osm, worldpop

logger = logging.getLogger(__name__)

_SEASON_BY_MONTH: Final[dict[int, str]] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

_DISTANCE_KINDS: Final = ("roads", "railways", "waterways")


def sub_point_distance_stats(
    spec: GridSpec, cell_id: npt.NDArray[np.int64], lines: gpd.GeoSeries, n_per_side: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Mean and std of distance-to-nearest-line, over `n_per_side**2` sub-points per cell.

    Local, CPU-side equivalent of the DEM sub-pixel `reduceResolution` in `sources/dem.py`:
    OSM geometry has no Earth Engine representation, so the aggregation happens here via a
    `shapely.STRtree` instead of server side.
    """
    n = len(cell_id)
    if lines.empty:
        nan = np.full(n, np.nan, dtype="float32")
        return nan, nan

    x, y = spec.sub_points(cell_id, n_per_side)
    tree = STRtree(lines.to_numpy())
    _, distance = tree.query_nearest(
        points(x.ravel(), y.ravel()), all_matches=False, return_distance=True
    )
    distance = distance.reshape(x.shape)
    return distance.mean(axis=1).astype("float32"), distance.std(axis=1).astype("float32")


def distance_frame(spec: GridSpec, active: npt.NDArray[np.int64], config: Config) -> pd.DataFrame:
    n = config.human.osm_distance_subpoints_per_side
    frame = pd.DataFrame({"cell_id": active})
    for kind in _DISTANCE_KINDS:
        lines = osm.fetch_lines(config, kind).geometry
        mean, std = sub_point_distance_stats(spec, active, lines, n)
        frame[f"dist_{kind}_mean"] = mean
        frame[f"dist_{kind}_std"] = std
    return frame


def poi_frame(spec: GridSpec, active: npt.NDArray[np.int64], config: Config) -> pd.DataFrame:
    """Recreational POI density and accommodation bed capacity, per active cell."""
    poi = osm.fetch_poi(config)
    cell_id = spec.points_to_cells(poi.geometry.x, poi.geometry.y)
    poi = poi.assign(cell_id=cell_id).loc[cell_id >= 0]

    is_recreational = poi["tourism"].isin(osm.RECREATIONAL_TOURISM_VALUES) | (
        poi["amenity"] == "parking"
    )
    recreational_count = poi.loc[is_recreational].groupby("cell_id").size()

    capacity = poi[list(osm.CAPACITY_TAG_PRIORITY)].bfill(axis=1).iloc[:, 0]
    is_accommodation = poi["tourism"].isin(osm.ACCOMMODATION_TOURISM_VALUES)
    accommodation_capacity = (
        capacity.where(is_accommodation).groupby(poi["cell_id"]).sum(min_count=1)
    )

    frame = pd.DataFrame({"cell_id": active}).set_index("cell_id")
    cell_km2 = (config.resolution_m / 1000) ** 2
    frame["recreational_poi_density"] = (
        recreational_count.reindex(frame.index, fill_value=0) / cell_km2
    )
    frame["accommodation_capacity"] = accommodation_capacity.reindex(frame.index, fill_value=0.0)
    return frame.reset_index().astype(
        {"recreational_poi_density": "float32", "accommodation_capacity": "float32"}
    )


def load_natura2000(config: Config) -> BaseGeometry:
    """The PAT Natura 2000 habitat layer as one geometry, in the working CRS."""
    path = config.path(config.paths.pat_natura2000)
    if not path.is_file():
        raise FileNotFoundError(f"Natura 2000 habitat layer not found: {path}")

    layer = gpd.read_file(path)
    verify_crs(layer.crs, config.crs, path)

    geometry: BaseGeometry = layer.geometry.union_all()
    logger.info("Natura 2000 from %s: %d feature(s)", path.name, len(layer))
    return geometry


def natura2000_frame(spec: GridSpec, active: npt.NDArray[np.int64], config: Config) -> pd.DataFrame:
    fraction = waterbody_fraction(spec, load_natura2000(config))
    return pd.DataFrame({"cell_id": active}).assign(
        natura2000_fraction=fraction[active].astype("float32"),
        natura2000=lambda f: f["natura2000_fraction"] >= config.human.natura2000_threshold,
    )


def population_frame(spec: GridSpec, active: npt.NDArray[np.int64], config: Config) -> pd.DataFrame:
    """One row per active cell and WorldPop year, mirroring `landcover.parquet`'s shape."""
    path = worldpop.fetch_population(spec, config)
    bands = read_cell_bands(spec, path, worldpop.band_names(config))

    frames = []
    for year in config.human.worldpop_years:
        frame = pd.DataFrame({"cell_id": np.arange(spec.n_cells, dtype="int32")})
        frame["worldpop_year"] = np.int16(year)
        frame["pop_density"] = bands[f"pop_density_{year}"].astype("float32")
        frames.append(frame.loc[frame["cell_id"].isin(active)])
    return pd.concat(frames, ignore_index=True)


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Month, season, weekday and Italian public holiday flag, for any date, no lookup.

    Unlike every other feature here, this is keyed by date, not by cell. Kept as a standalone
    function rather than an `EXTRACTORS` entry.
    """
    index = pd.DatetimeIndex(dates)
    it_holidays = holidays.country_holidays(
        "IT", years=range(index.year.min(), index.year.max() + 1)
    )

    return pd.DataFrame(
        {
            "date": index,
            "month": index.month.astype("int8"),
            "season": pd.Categorical(index.month.map(_SEASON_BY_MONTH)),
            "day_of_week": index.dayofweek.astype("int8"),
            "is_weekend": index.dayofweek >= 5,
            "is_holiday": [d in it_holidays for d in index.date],
        }
    )


def validate_human(human: pd.DataFrame, population: pd.DataFrame, config: Config) -> None:
    """Check plausibility, logging loudly."""
    for kind in _DISTANCE_KINDS:
        column = f"dist_{kind}_mean"
        negative = int((human[column] < 0).sum())
        if negative:
            logger.error("%d cell(s) have a negative %s", negative, column)

    if not human["natura2000_fraction"].between(0, 1).all():
        logger.error("natura2000_fraction outside [0, 1]")
    logger.info(
        "Natura 2000: %d / %d cell(s) flagged at threshold %.2f",
        int(human["natura2000"].sum()),
        len(human),
        config.human.natura2000_threshold,
    )

    zero_capacity = int((human["accommodation_capacity"] == 0).sum())
    logger.info(
        "Accommodation: %d / %d cell(s) have zero capacity (OSM beds/capacity/rooms tagging "
        "is incomplete; not imputed)",
        zero_capacity,
        len(human),
    )

    high = population.loc[population["pop_density"] > config.human.max_pop_density_km2]
    if not high.empty:
        logger.error(
            "%d (cell, year) pair(s) exceed the %.0f/km2 plausibility ceiling",
            len(high),
            config.human.max_pop_density_km2,
        )
    logger.info(
        "WorldPop: years %s, mean %.1f/km2, max %.1f/km2",
        config.human.worldpop_years,
        population["pop_density"].mean(),
        population["pop_density"].max(),
    )


def extract_human(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `human.parquet` (static, one row per active cell) and `human_population.parquet`
    (one row per active cell and WorldPop year).
    """
    out = config.path(config.paths.human_out)
    population_out = config.path(config.paths.human_population_out)
    if out.exists() and population_out.exists() and not force:
        logger.info("Outputs already exist, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    spec, grid = load_grid(config)
    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()

    human = (
        distance_frame(spec, active, config)
        .merge(poi_frame(spec, active, config), on="cell_id")
        .merge(natura2000_frame(spec, active, config), on="cell_id")
    )
    population = population_frame(spec, active, config)
    validate_human(human, population, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    human.to_parquet(out, index=False)
    population.to_parquet(population_out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(human), human.shape[1], out)
    logger.info("Wrote %d rows to %s", len(population), population_out)
    return human
