"""PAT fire cadastre parsing, timestamps and ignition points.

Turns the official Servizio Foreste fire inventory (`Incendi__date.shp`) into the
positive-sample table the rest of the pipeline attaches to.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd

from tfire.config import Config
from tfire.geo import verify_crs

logger = logging.getLogger(__name__)

# Source column names in the PAT shapefile.
COL_OBJECTID: Final = "objectid"
COL_AREA_HA: Final = "area_ha"
COL_CAUSE: Final = "cause"
COL_LOC: Final = "loc"
COL_ANNO_INC: Final = "anno_inc"
COL_DURATA_MIN: Final = "durata_min"

START_FIELDS: Final = ("gg_inizio", "mm_inizio", "aa_inizio", "ora_inizio", "min_inizio")
END_FIELDS: Final = ("gg_fine", "mm_fine", "aa_fine", "ora_fine", "min_fine")

# The 13 anomalous rows carry these two shapes instead of a bare number.
_FULL_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")
_FULL_TIME = re.compile(r"^\s*(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\s*$")

_MAX_HOUR: Final = 23
_MAX_MINUTE: Final = 59


class Component(StrEnum):
    """Which scalar a split date/time column is supposed to hold."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    HOUR = "hour"
    MINUTE = "minute"


_DATE_GROUP: Final[dict[Component, int]] = {
    Component.DAY: 1,
    Component.MONTH: 2,
    Component.YEAR: 3,
}
_TIME_GROUP: Final[dict[Component, int]] = {
    Component.HOUR: 1,
    Component.MINUTE: 2,
}


def normalize_component(series: pd.Series, component: Component) -> tuple[pd.Series, pd.Series]:
    """Coerce a split date/time column to numbers, repairing full date/time strings."""
    text = series.astype("string").str.strip()

    values = pd.to_numeric(text, errors="coerce").astype("Float64")
    repaired = pd.Series(False, index=series.index, dtype=bool)

    pattern, group_map = (
        (_FULL_DATE, _DATE_GROUP) if component in _DATE_GROUP else (_FULL_TIME, _TIME_GROUP)
    )
    group = group_map.get(component)
    if group is None:
        # e.g. asking for MINUTE from a date-shaped string: nothing to recover.
        return values, repaired

    extracted = text.str.extract(pattern, expand=True)
    hits = extracted[group - 1].notna()
    if hits.any():
        values = values.mask(hits, pd.to_numeric(extracted[group - 1], errors="coerce"))
        repaired = repaired | hits.fillna(False)

    return values.astype("Float64"), repaired


def normalize_fields(
    frame: pd.DataFrame, fields: tuple[str, ...], components: tuple[Component, ...]
) -> tuple[dict[Component, pd.Series], pd.Series]:
    """Normalize a group of five split columns (day, month, year, hour, minute)."""
    values: dict[Component, pd.Series] = {}
    repaired = pd.Series(False, index=frame.index, dtype=bool)

    for field, component in zip(fields, components, strict=True):
        column, was_repaired = normalize_component(frame[field], component)
        values[component] = column
        repaired = repaired | was_repaired
        if was_repaired.any():
            logger.info("Repaired %d value(s) in column %r", int(was_repaired.sum()), field)

    return values, repaired


def _clamp(series: pd.Series, maximum: int) -> pd.Series:
    """Blank out-of-range time components so they cannot invalidate a valid date."""
    numeric = pd.to_numeric(series, errors="coerce")
    out_of_range = numeric.notna() & ((numeric < 0) | (numeric > maximum))
    if bool(out_of_range.any()):
        logger.warning(
            "%d time component(s) outside [0, %d]; treated as 0", int(out_of_range.sum()), maximum
        )
    return numeric.mask(out_of_range, 0).fillna(0)


def build_datetime(
    day: pd.Series,
    month: pd.Series,
    year: pd.Series,
    hour: pd.Series | None = None,
    minute: pd.Series | None = None,
) -> pd.Series:
    """Assemble a timestamp from split components, yielding NaT instead of raising.

    Missing or non-numeric day/month/year gives NaT. Impossible calendar dates
    (31 February) give NaT. Missing hour/minute default to midnight, since a date with
    an unknown time is still usable and the time is discarded downstream anyway.
    """
    index = day.index
    zero = pd.Series(0.0, index=index)

    d = pd.to_numeric(day, errors="coerce")
    m = pd.to_numeric(month, errors="coerce")
    y = pd.to_numeric(year, errors="coerce")
    h = _clamp(hour, _MAX_HOUR) if hour is not None else zero
    mi = _clamp(minute, _MAX_MINUTE) if minute is not None else zero

    def as_text(series: pd.Series, width: int) -> pd.Series:
        return series.astype("Int64").astype("string").str.zfill(width)

    # Build "YYYY-MM-DD HH:MM" and let pandas reject impossible dates. The string form
    # is used rather than pd.to_datetime(DataFrame) because the latter raises on NaN
    # components in some pandas versions even under errors="coerce".
    stamp = (
        as_text(y, 4)
        + "-"
        + as_text(m, 2)
        + "-"
        + as_text(d, 2)
        + " "
        + as_text(h, 2)
        + ":"
        + as_text(mi, 2)
    )

    return pd.to_datetime(stamp, format="%Y-%m-%d %H:%M", errors="coerce")


def flag_near_midnight(hour: pd.Series, config: Config) -> pd.Series:
    """`hour >= 22 or hour < 2`: timestamps most exposed to day-attribution error."""
    numeric = pd.to_numeric(hour, errors="coerce")
    flagged = (numeric >= config.fires.near_midnight_start_hour) | (
        numeric < config.fires.near_midnight_end_hour
    )
    return flagged.fillna(False).astype(bool)


def flag_suspicious_default_time(hour: pd.Series, config: Config) -> pd.Series:
    """`0 <= hour <= 2`: the anomalous spike of likely placeholder times."""
    numeric = pd.to_numeric(hour, errors="coerce")
    flagged = (numeric >= 0) & (numeric <= config.fires.suspicious_time_max_hour)
    return flagged.fillna(False).astype(bool)


def load_fire_polygons(path: Path, expected_crs: str) -> gpd.GeoDataFrame:
    """Read the cadastre and assert its CRS. Never reprojects."""
    if not path.is_file():
        raise FileNotFoundError(f"Fire shapefile not found: {path}")

    gdf = gpd.read_file(path)
    logger.info("Loaded %d polygons from %s", len(gdf), path)

    verify_crs(gdf.crs, expected_crs, path)

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        logger.warning("%d geometry/geometries are topologically invalid", invalid)

    return gdf


def extract_ignition_points(geometries: gpd.GeoSeries) -> tuple[gpd.GeoSeries, pd.Series]:
    """Pick one interior point per polygon: centroid, else `representative_point`."""
    centroids = geometries.centroid
    inside = geometries.contains(centroids).to_numpy(dtype=bool)
    fallback = geometries.representative_point()

    points = gpd.GeoSeries(
        np.where(inside, centroids.to_numpy(), fallback.to_numpy()),
        index=geometries.index,
        crs=geometries.crs,
    )
    method = pd.Series(
        np.where(inside, "centroid", "representative_point"),
        index=geometries.index,
        dtype="string",
    )

    outside = ~geometries.contains(points).to_numpy(dtype=bool)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} ignition point(s) fall outside their own polygon "
            f"at positional index/indices {np.flatnonzero(outside).tolist()[:10]}"
        )

    return points, method


def validate_fires(fires: pd.DataFrame, config: Config) -> None:
    """Check the acceptance criteria, logging loudly"""
    expected_rows = config.fires.expected_row_count
    n_rows = len(fires)
    if n_rows != expected_rows:
        logger.error("Row count is %d, expected %d", n_rows, expected_rows)
    else:
        logger.info("Row count %d matches expectation", n_rows)

    n_valid = int(fires["ignition_date"].notna().sum())
    if n_valid < config.fires.min_valid_timestamps:
        logger.error(
            "Only %d valid ignition dates, expected at least %d",
            n_valid,
            config.fires.min_valid_timestamps,
        )
    else:
        logger.info(
            "Valid ignition dates: %d / %d (minimum %d)",
            n_valid,
            n_rows,
            config.fires.min_valid_timestamps,
        )

    duplicated = int(fires["fire_id"].duplicated().sum())
    if duplicated:
        logger.error("%d duplicated fire_id value(s)", duplicated)

    n_normalized = int(fires["date_format_normalized"].sum())
    if n_normalized != config.fires.expected_normalized_rows:
        logger.warning(
            "%d row(s) needed date-format repair, expected %d. The provider may have "
            "changed the field format again; re-check normalize_component()",
            n_normalized,
            config.fires.expected_normalized_rows,
        )
    else:
        logger.info("Date-format repairs: %d (as expected)", n_normalized)

    if n_valid:
        logger.info(
            "Ignition date range: %s -> %s",
            fires["ignition_date"].min().date(),
            fires["ignition_date"].max().date(),
        )
    logger.info(
        "Flags: near_midnight=%d (%.2f%%), suspicious_default_time=%d",
        int(fires["near_midnight"].sum()),
        100 * float(fires["near_midnight"].mean()),
        int(fires["suspicious_default_time"].sum()),
    )
    logger.info(
        "Ignition point method: %s",
        fires["ignition_point_method"].value_counts().to_dict(),
    )


def build_fires_table(
    gdf: gpd.GeoDataFrame, config: Config
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Build the positives table and the companion polygon table."""
    components = (Component.DAY, Component.MONTH, Component.YEAR, Component.HOUR, Component.MINUTE)

    start, start_repaired = normalize_fields(gdf, START_FIELDS, components)
    end, end_repaired = normalize_fields(gdf, END_FIELDS, components)

    start_datetime = build_datetime(
        start[Component.DAY],
        start[Component.MONTH],
        start[Component.YEAR],
        start[Component.HOUR],
        start[Component.MINUTE],
    )
    end_datetime = build_datetime(
        end[Component.DAY],
        end[Component.MONTH],
        end[Component.YEAR],
        end[Component.HOUR],
        end[Component.MINUTE],
    )

    points, method = extract_ignition_points(gdf.geometry)

    fire_id = pd.to_numeric(gdf[COL_OBJECTID], errors="coerce").astype("Int64")
    start_hour = start[Component.HOUR].astype("Int64")

    fires = pd.DataFrame(
        {
            "fire_id": fire_id,
            "x": points.x.to_numpy(),
            "y": points.y.to_numpy(),
            # audit fields only, never join features on start/end time
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "ignition_date": start_datetime.dt.normalize(),
            "start_hour": start_hour,
            "area_ha": pd.to_numeric(gdf[COL_AREA_HA], errors="coerce").astype(float),
            "cause": pd.to_numeric(gdf[COL_CAUSE], errors="coerce").astype("Int64"),
            "loc": gdf[COL_LOC].astype("string"),
            "anno_inc": pd.to_numeric(gdf[COL_ANNO_INC], errors="coerce").astype("Int64"),
            "durata_min": pd.to_numeric(gdf[COL_DURATA_MIN], errors="coerce").astype("Int64"),
            "near_midnight": flag_near_midnight(start[Component.HOUR], config),
            "suspicious_default_time": flag_suspicious_default_time(start[Component.HOUR], config),
            "ignition_point_method": method,
            "date_format_normalized": (start_repaired | end_repaired).astype(bool),
        },
        index=gdf.index,
    ).reset_index(drop=True)

    polygons = gpd.GeoDataFrame(
        {"fire_id": fires["fire_id"].to_numpy()},
        geometry=gdf.geometry.to_numpy(),
        crs=gdf.crs,
    )

    return fires, polygons


def build_positives(config: Config, force: bool = False) -> pd.DataFrame:
    """Parse the cadastre shapefile into `fires.parquet` and `fire_polygons.parquet`."""
    fires_out = config.path(config.paths.fires_out)
    polygons_out = config.path(config.paths.fire_polygons_out)

    if fires_out.exists() and polygons_out.exists() and not force:
        logger.info("Outputs already exist, skipping (use --force to rebuild): %s", fires_out)
        return pd.read_parquet(fires_out)

    gdf = load_fire_polygons(config.path(config.paths.fires_shapefile), config.crs)
    fires, polygons = build_fires_table(gdf, config)
    validate_fires(fires, config)

    fires_out.parent.mkdir(parents=True, exist_ok=True)
    fires.to_parquet(fires_out, index=False)
    polygons.to_parquet(polygons_out, index=False)
    logger.info("Wrote %d rows to %s", len(fires), fires_out)
    logger.info("Wrote %d geometries to %s", len(polygons), polygons_out)

    return fires
