"""Where a cell is: its center coordinates and whether it holds standing water."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from tfire.config import Config
from tfire.geo import verify_crs
from tfire.grid import GridSpec, load_grid

logger = logging.getLogger(__name__)


def load_waterbodies(config: Config) -> BaseGeometry:
    """The PAT standing-water layer as one geometry, in the working CRS."""
    path = config.path(config.paths.pat_waterbodies)
    if not path.is_file():
        raise FileNotFoundError(f"PAT water bodies not found: {path}")

    layer = gpd.read_file(path)
    verify_crs(layer.crs, config.crs, path)

    geometry: BaseGeometry = layer.geometry.union_all()
    logger.info(
        "Water bodies from %s: %d feature(s), %.1f km2", path.name, len(layer), geometry.area / 1e6
    )
    return geometry


def waterbody_fraction(spec: GridSpec, waterbodies: BaseGeometry) -> npt.NDArray[np.float64]:
    """Per-cell share of the cell area covered by standing water."""
    fraction = np.zeros(spec.n_cells, dtype=np.float64)

    covered = spec.polygon_to_cells(waterbodies)
    if not covered:
        return fraction

    boxes = spec.cell_polygons(covered).to_numpy()
    area = shapely.area(shapely.intersection(waterbodies, boxes))
    fraction[covered] = area / spec.resolution_m**2
    return fraction


def build_geography_frame(grid: pd.DataFrame, spec: GridSpec, config: Config) -> pd.DataFrame:
    fraction = waterbody_fraction(spec, load_waterbodies(config))
    active = grid.loc[grid["is_trentino"]]

    return pd.DataFrame(
        {
            "cell_id": active["cell_id"].to_numpy(),
            "x_coordinate": active["x_coordinate"].to_numpy(),
            "y_coordinate": active["y_coordinate"].to_numpy(),
            "waterbody_fraction": fraction[active["cell_id"].to_numpy()].astype("float32"),
        }
    ).assign(
        is_waterbody=lambda frame: (
            frame["waterbody_fraction"] >= config.geography.waterbody_threshold
        )
    )


def validate_geography(geography: pd.DataFrame, config: Config) -> None:
    """Check the counts the later stages read, logging loudly."""
    fraction = geography["waterbody_fraction"]
    if not fraction.between(0, 1).all():
        logger.error(
            "waterbody_fraction outside [0, 1]: min %.3f, max %.3f", fraction.min(), fraction.max()
        )

    logger.info(
        "Water: %.1f km2 over %d active cell(s), %d flagged is_waterbody at threshold %.2f",
        float(fraction.sum()) * config.resolution_m**2 / 1e6,
        int((fraction > 0).sum()),
        int(geography["is_waterbody"].sum()),
        config.geography.waterbody_threshold,
    )


def extract_geography(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `geography.parquet`, one row per active cell."""
    out = config.path(config.paths.geography_out)
    if out.exists() and not force:
        logger.info("Output already exists, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    spec, grid = load_grid(config)
    geography = build_geography_frame(grid, spec, config)
    validate_geography(geography, config)

    out.parent.mkdir(parents=True, exist_ok=True)
    geography.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d columns to %s", len(geography), geography.shape[1], out)
    return geography
