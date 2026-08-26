"""Everything the API reads: risk maps, overlays, cell lookups and the numbers behind the site."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.api.state import active_version
from tfire.config import Config
from tfire.grid import GridSpec, load_boundary, load_grid
from tfire.inference import (
    DRIVER_COLUMNS,
    PROBABILITY_BAND,
    RANK_BAND,
    FeatureUnavailableError,
    active_cells,
    cached_span,
    model_fingerprint,
    predict_days,
    table_last_date,
)
from tfire.models.danger import DangerClasses, summarize
from tfire.raster import grid_transform, read_cell_bands
from tfire.sources.bias import bias_fingerprint

if TYPE_CHECKING:
    import geopandas as gpd
    from pyproj import Transformer

logger = logging.getLogger(__name__)

T = TypeVar("T")

WEB_MERCATOR: Final = "EPSG:3857"
WGS84: Final = "EPSG:4326"

# green, amber, orange, then the two reds
CLASS_COLORS: Final = ("#2f7d32", "#ffc107", "#fd7e14", "#c6020f", "#720000")

RANK_COLORS: Final = ("#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d")

PALETTES: Final = {"classes": PROBABILITY_BAND, "rank": RANK_BAND}

_locks: dict[date, threading.Lock] = {}
_locks_guard = threading.Lock()
_memo: dict[str, Any] = {}
# reentrant: one memoized value is built out of another, `overlay_geometry` off `grid_spec`
_memo_guard = threading.RLock()


class DateOutOfRangeError(Exception):
    """A date no provider will ever cover, as opposed to one that merely failed today."""


def _cached(key: str, build: Callable[[], T]) -> T:
    """Process-lifetime memo for the tables that do not change while the service runs."""
    with _memo_guard:
        if key not in _memo:
            _memo[key] = build()
        value: T = _memo[key]
        return value


def _stamp(path: Path) -> tuple[float, int] | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_mtime, info.st_size


def _cached_file(key: str, path: Path, build: Callable[[], T]) -> T:
    """Memo keyed on the source file, so a rebuild is picked up without a restart.

    The admin surface rebuilds the grid and the static tables from inside the running
    process. Held under a bare name these would stay in memory for the life of the worker,
    and the service would answer about a grid that no longer exists.
    """
    stamp = _stamp(path)
    with _memo_guard:
        entry = _memo.get(key)
        if entry is None or entry[0] != stamp:
            _memo[key] = (stamp, build())
        value: T = _memo[key][1]
        return value


def _grid_cached(config: Config, key: str, build: Callable[[Config], T]) -> T:
    """Memo for everything derived from the grid parquet, keyed on that file."""
    return _cached_file(key, config.path(config.paths.grid_out), lambda: build(config))


class StaleArtifactError(Exception):
    """An artifact on disk that belongs to a different model or a different grid."""


def danger_classes(config: Config) -> DangerClasses:
    """The active version's class breaks, refused rather than served when they do not fit."""
    from tfire.models.danger import reference_days

    version = active_version(config)

    def build() -> DangerClasses:
        classes = DangerClasses.read(_danger_path(config))
        if classes.model_version != version:
            raise StaleArtifactError(
                f"danger_classes.json was cut for model {classes.model_version}, "
                f"and {version} is being served. Run `tfire danger-classes --force`."
            )

        expected = config.grid.expected_active_cells * len(reference_days(config))
        if classes.rows != expected:
            raise StaleArtifactError(
                f"danger_classes.json was cut on {classes.rows:,} reference rows, and the "
                f"active grid over the reference days is {expected:,}. "
                "Run `tfire danger-classes --force`."
            )

        if classes.config_sha256 != config.digest():
            logger.warning(
                "danger_classes.json was cut under a different config; breaks may not match "
                "the settings this service runs on"
            )
        return classes

    return _cached_file(f"danger:{version}", _danger_path(config), build)


def _danger_path(config: Config) -> Path:
    from tfire.inference import model_directory
    from tfire.models.danger import DANGER_FILENAME

    return model_directory(config) / DANGER_FILENAME


def _day_lock(day: date) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(day, threading.Lock())


def outputs(config: Config, day: date) -> dict[str, Path]:
    directory = config.path(config.paths.risk_dir)
    return {
        "raster": directory / f"risk_{day:%Y-%m-%d}.tif",
        "table": directory / f"risk_{day:%Y-%m-%d}.parquet",
        "meta": directory / f"risk_{day:%Y-%m-%d}.json",
    }


def date_bounds(config: Config, today: date | None = None) -> tuple[date, date]:
    stamp = today or date.today()
    return config.date_range.start, stamp + timedelta(days=config.forecast.horizon_days)


def is_stale(config: Config, day: date) -> bool:
    """Whether a map on disk still describes what this service would produce today."""
    import pyarrow.parquet as pq

    paths = outputs(config, day)
    if not all(path.is_file() for path in paths.values()):
        return True

    sidecar = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if sidecar.get("model_version") != active_version(config):
        return True
    if sidecar.get("model_fingerprint") != model_fingerprint(config):
        return True
    if sidecar.get("cells") != len(_grid_cached(config, "active_cells", active_cells)):
        return True
    if _served_remotely(config, day, sidecar):
        return True

    columns = set(pq.read_schema(paths["table"]).names)
    return not set(DRIVER_COLUMNS).issubset(columns)


def _served_remotely(config: Config, day: date, sidecar: dict[str, Any]) -> bool:
    """Whether a map was scored off a weather provider for a date the backbone now covers."""
    start, end = cached_span(config)
    if start <= day <= end:
        return sidecar.get("sources") != ["cached"]

    # past the backbone the map is remote, and the wind correction is part of how it was made
    return sidecar.get("bias_map") != bias_fingerprint(config)


def ensure_day(config: Config, day: date, allow_compute: bool = True) -> dict[str, Path]:
    """The risk map for a date, computed if it is absent or stale."""
    first, last = date_bounds(config)
    if not first <= day <= last:
        raise DateOutOfRangeError(f"{day} is outside {first} to {last}")

    if not is_stale(config, day):
        return outputs(config, day)
    if not allow_compute:
        raise FeatureUnavailableError(
            day, ["risk"], "no risk map is stored for this date and on-demand scoring is off."
        )

    with _day_lock(day):
        if is_stale(config, day):
            predict_days(config, day, days=1, force=True)
        return outputs(config, day)


def read_table(config: Config, day: date) -> pd.DataFrame:
    return pd.read_parquet(outputs(config, day)["table"])


def read_sidecar(config: Config, day: date) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(outputs(config, day)["meta"].read_text(encoding="utf-8"))
    return payload


def grid_spec(config: Config) -> GridSpec:
    spec: GridSpec = _grid_cached(config, "grid_spec", lambda cfg: load_grid(cfg)[0])
    return spec


@dataclass(frozen=True)
class OverlayGeometry:
    """Where the warped PNG sits, and how big it is."""

    width: int
    height: int
    transform: Any
    bounds_wgs84: tuple[float, float, float, float]

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        west, south, east, north = self.bounds_wgs84
        return [[south, west], [north, east]]


def overlay_geometry(config: Config) -> OverlayGeometry:
    """The EPSG:3857 raster the overlay is drawn into, fixed for the life of the grid.

    Upsampled towards `overlay_max_px` before warping: at one pixel per cell the reprojection
    would drop or double whole cells, and a 500 m cell is the unit a reader is being shown.
    """

    def build() -> OverlayGeometry:
        from rasterio.transform import from_bounds
        from rasterio.warp import transform_bounds

        spec = grid_spec(config)
        bounds = transform_bounds(spec.crs, WEB_MERCATOR, *spec.bounds)
        scale = max(1, config.app.overlay_max_px // max(spec.n_cols, spec.n_rows))
        width, height = spec.n_cols * scale, spec.n_rows * scale
        return OverlayGeometry(
            width=width,
            height=height,
            transform=from_bounds(*bounds, width, height),
            bounds_wgs84=transform_bounds(WEB_MERCATOR, WGS84, *bounds),
        )

    geometry: OverlayGeometry = _grid_cached(config, "overlay_geometry", lambda _cfg: build())
    return geometry


def _rgba(
    values: npt.NDArray[np.float64],
    classes: DangerClasses,
    spec: GridSpec,
    colors: Sequence[str],
) -> npt.NDArray[np.uint8]:
    assigned = classes.classify(values).reshape(spec.n_rows, spec.n_cols)
    image = np.zeros((4, spec.n_rows, spec.n_cols), dtype="uint8")
    for index, color in enumerate(colors[: len(classes.class_keys)]):
        mask = assigned == index
        red, green, blue = (int(color[start : start + 2], 16) for start in (1, 3, 5))
        image[0][mask], image[1][mask], image[2][mask] = red, green, blue
        image[3][mask] = 255
    return image


# the absolute scale is cut where fire danger is genuinely rare, which within a single day
# would leave every cell in one class. This palette answers a different question, where the
# worst cells of *this* day are, so it keeps breaks that split a day into readable parts.
RANK_PERCENTILES: Final = (50.0, 80.0, 95.0, 99.0)


def _rank_classes(classes: DangerClasses) -> DangerClasses:
    """The same five colors, cut on the day's own ranking instead of the record's."""
    return DangerClasses(
        breaks=[value / 100 for value in RANK_PERCENTILES],
        percentiles=list(RANK_PERCENTILES),
        class_keys=classes.class_keys,
        reference="within-day percentile",
        reference_years=classes.reference_years,
        rows=0,
        model_version=classes.model_version,
        config_sha256=classes.config_sha256,
    )


def overlay_token(config: Config, classes: DangerClasses) -> str:
    """Identity of what an overlay draws, for the URL.

    The path alone is not enough: anything that changes the picture without changing the
    address would stay on screen forever, because the responses are served `immutable`.
    """
    return f"{active_version(config)}-{model_fingerprint(config)[:8]}-{_visual_digest(classes)}"


def _visual_digest(classes: DangerClasses) -> str:
    """Short fingerprint of everything that decides a pixel's color.

    Thresholds alone are not enough: repainting a palette or moving a rank break changes
    every overlay while leaving the breaks where they were.
    """
    payload = "|".join(
        (
            ",".join(f"{value:.6e}" for value in classes.breaks),
            ",".join(f"{value:.6e}" for value in RANK_PERCENTILES),
            ",".join(CLASS_COLORS),
            ",".join(RANK_COLORS),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def overlay_path(config: Config, day: date, palette: str, classes: DangerClasses) -> Path:
    stem = (
        f"risk_{day:%Y-%m-%d}_{palette}_{active_version(config)}"
        f"_{model_fingerprint(config)[:8]}_{_visual_digest(classes)}"
    )
    return config.path(config.paths.overlay_dir) / f"{stem}.png"


def render_overlay(config: Config, day: date, palette: str) -> Path:
    """The day's risk as a PNG on the Web Mercator lattice Leaflet draws in."""
    from PIL import Image
    from rasterio.warp import Resampling, reproject

    if palette not in PALETTES:
        raise ValueError(f"Unknown palette {palette!r}, have {sorted(PALETTES)}")

    classes = danger_classes(config)
    path = overlay_path(config, day, palette, classes)
    if path.is_file():
        return path

    spec = grid_spec(config)
    band = read_cell_bands(spec, outputs(config, day)["raster"])[PALETTES[palette]]
    absolute = palette == "classes"
    source = _rgba(
        band,
        classes if absolute else _rank_classes(classes),
        spec,
        CLASS_COLORS if absolute else RANK_COLORS,
    )

    geometry = overlay_geometry(config)
    warped = np.zeros((4, geometry.height, geometry.width), dtype="uint8")
    reproject(
        source,
        warped,
        src_transform=grid_transform(spec),
        src_crs=spec.crs,
        dst_transform=geometry.transform,
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(warped, 0, -1), mode="RGBA").save(path, optimize=True)
    logger.info("Rendered %s (%dx%d)", path.name, geometry.width, geometry.height)
    return path


def day_summary(config: Config, day: date) -> dict[str, Any]:
    table = read_table(config, day)
    classes = danger_classes(config)
    probability = table[PROBABILITY_BAND].to_numpy("float64")

    top = table.nlargest(10, PROBABILITY_BAND)
    cell_ids = top["cell_id"].to_numpy("int64")
    ranked = top[PROBABILITY_BAND].to_numpy("float64")
    within_day = top[RANK_BAND].to_numpy("float64")
    assigned = classes.classify(ranked)

    x, y = grid_spec(config).cell_center(cell_ids)
    lon, lat = _to_wgs84(config, x, y)

    return {
        "date": day.isoformat(),
        "provenance": read_sidecar(config, day),
        "classes": summarize(classes, probability),
        "probability": {
            "mean": float(np.mean(probability)),
            "max": float(np.max(probability)),
        },
        "top_cells": [
            {
                "cell_id": int(cell_ids[index]),
                "lon": float(lon[index]),
                "lat": float(lat[index]),
                "probability": float(ranked[index]),
                "rank": float(within_day[index]),
                "danger_class": int(assigned[index]),
            }
            for index in range(len(cell_ids))
        ],
    }


def _transformer(config: Config, forward: bool) -> Transformer:
    from pyproj import Transformer

    key = f"transformer_{'to_grid' if forward else 'to_wgs84'}"
    source, target = (WGS84, config.crs) if forward else (config.crs, WGS84)
    return _cached(key, lambda: Transformer.from_crs(source, target, always_xy=True))


def _to_wgs84(config: Config, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[Any, Any]:
    return _transformer(config, forward=False).transform(x, y)


def query_cell(config: Config, day: date, lon: float, lat: float) -> dict[str, Any] | None:
    """The clicked cell, or `None` when the point is outside the modeled province."""
    x, y = _transformer(config, forward=True).transform(lon, lat)
    cell_id = grid_spec(config).point_to_cell(float(x), float(y))
    if cell_id is None:
        return None

    table = read_table(config, day)
    row = table[table["cell_id"] == cell_id]
    if row.empty:
        return None

    classes = danger_classes(config)
    probability = float(row[PROBABILITY_BAND].iloc[0])
    center_x, center_y = grid_spec(config).cell_center([cell_id])
    center_lon, center_lat = _to_wgs84(config, center_x, center_y)

    topography = _cached_file(
        "topography",
        config.path(config.paths.topography_out),
        lambda: pd.read_parquet(
            config.path(config.paths.topography_out), columns=["cell_id", "elevation_mean"]
        ).set_index("cell_id"),
    )

    drivers = {name: _finite(row[name].iloc[0]) for name in DRIVER_COLUMNS}
    return {
        "cell_id": cell_id,
        "lon": float(center_lon[0]),
        "lat": float(center_lat[0]),
        "elevation_m": _finite(topography["elevation_mean"].get(cell_id, np.nan)),
        "probability": probability,
        "rank": float(row[RANK_BAND].iloc[0]),
        "danger_class": int(classes.classify([probability])[0]),
        "record_percentile": classes.record_percentile(probability),
        "drivers": drivers,
    }


def _finite(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return None if np.isnan(number) else number


def fires_on(config: Config, day: date) -> dict[str, Any]:
    """The cadastre polygons that burned on a date, as GeoJSON in WGS84."""
    import geopandas as gpd

    def build() -> gpd.GeoDataFrame:
        fires = pd.read_parquet(
            config.path(config.paths.fires_out),
            columns=["fire_id", "ignition_date", "area_ha", "loc", "start_hour"],
        )
        polygons = gpd.read_parquet(config.path(config.paths.fire_polygons_out))
        return polygons.merge(fires, on="fire_id", how="inner").to_crs(WGS84)

    if not config.path(config.paths.fire_polygons_out).is_file():
        return {"type": "FeatureCollection", "features": []}

    frame = _cached_file("fire_polygons", config.path(config.paths.fire_polygons_out), build)
    matching = frame[frame["ignition_date"] == pd.Timestamp(day)]
    payload: dict[str, Any] = json.loads(matching.to_json(drop_id=True, default=str))
    return payload


def boundary_geojson(config: Config) -> dict[str, Any]:
    import geopandas as gpd

    def build() -> dict[str, Any]:
        # simplified in metres before reprojection: the outline is drawn over 500 m cells, so
        # half a cell of tolerance costs nothing visible and a fifth of the payload
        outline = gpd.GeoSeries([load_boundary(config)], crs=config.crs)
        payload: dict[str, Any] = json.loads(
            outline.simplify(config.resolution_m / 2).to_crs(WGS84).to_json()
        )
        return payload

    geojson: dict[str, Any] = _cached_file(
        "boundary", config.path(config.paths.pat_boundary), build
    )
    return geojson


def _table_last_date(config: Config, relative: Path, key: str) -> str | None:
    path = config.path(relative)
    stamp = _cached_file(key, path, lambda: table_last_date(path))
    return None if stamp is None else stamp.isoformat()


def _newest_composite(config: Config) -> str | None:
    """The latest cached Landsat month, which is what inference builds vegetation from.

    Not `vegetation_out`: that table feeds the training build, and reporting its end date would
    tell an operator the vegetation is current when the composites a served map actually uses
    stopped months earlier. Read off the newest raster alone rather than through `cached_months`,
    which opens all 177 of them, because this is polled while a job runs.
    """
    import rasterio

    from tfire.sources.landsat import parse_band_name

    rasters = sorted(config.path(config.paths.landsat_raw).glob("*.tif"))
    if not rasters:
        return None
    with rasterio.open(rasters[-1]) as source:
        months = [parse_band_name(name)[1] for name in source.descriptions]
    return max(months).isoformat() if months else None


def status(config: Config) -> dict[str, Any]:
    """Data freshness and what the service is currently serving with."""
    directory = config.path(config.paths.risk_dir)
    stored = sorted(path.stem.split("_", 1)[1] for path in directory.glob("risk_*.tif"))
    version = active_version(config)
    metrics_path = config.path(config.paths.trentino_model_dir) / version / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}

    return {
        "model_version": version,
        "config_matches_training": metrics.get("config_sha256") == config.digest(),
        "holdout_auprc": metrics.get("models", {})
        .get("xgboost", {})
        .get("holdout", {})
        .get("auprc"),
        "meteo_through": _table_last_date(config, config.paths.meteo_out, "meteo_last"),
        "fwi_through": _table_last_date(config, config.paths.fwi_out, "fwi_last"),
        "vegetation_through": _newest_composite(config),
        "risk_maps": {
            "count": len(stored),
            "first": stored[0] if stored else None,
            "last": stored[-1] if stored else None,
            "bytes": sum(path.stat().st_size for path in directory.glob("risk_*.*")),
        },
    }


def about(config: Config) -> dict[str, Any]:
    """The figures the project page reports, read from the artifacts that produced them."""
    version = active_version(config)
    directory = config.path(config.paths.trentino_model_dir) / version

    def read(path: Path) -> dict[str, Any]:
        payload: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )
        return payload

    evaluation = read(directory / "evaluation.json")
    metrics = read(directory / "metrics.json")
    quality = read(config.path(config.paths.quality_report_out))
    models = metrics.get("models", {})

    return {
        "record": {
            "start_year": config.date_range.start.year,
            "end_year": config.date_range.end.year,
            "cells": config.grid.expected_active_cells,
            "resolution_m": config.resolution_m,
            "positives": quality.get("positives"),
            "cadastre_ignitions": _cadastre_ignitions(config),
            "rows": quality.get("rows"),
            "features": quality.get("features"),
        },
        "model": {
            "version": version,
            "train_years": metrics.get("split", {}).get("train_years"),
            "test_years": metrics.get("split", {}).get("test_years"),
            "holdout": {
                name: models.get(name, {}).get("holdout", {}).get("auprc")
                for name in ("xgboost", "random_forest", "logistic", "fwi_only")
            },
            "pooled_out_of_fold_auprc": evaluation.get("pooled_out_of_fold", {}).get("auprc"),
            "spatial_cv_auprc": evaluation.get("spatial_cv", {})
            .get("pooled_out_of_fold", {})
            .get("auprc"),
            "attribution": evaluation.get("attribution", {}).get("per_temporal", []),
        },
        "sources": _source_families(config),
        "figures": ["pr_curves.png", "shap_categories.png"],
    }


def _cadastre_ignitions(config: Config) -> int | None:
    """Ignitions on record inside the modeling window, not the positives that were sampled."""
    path = config.path(config.paths.fires_out)
    if not path.is_file():
        return None

    def build() -> int:
        dates = pd.read_parquet(path, columns=["ignition_date"])["ignition_date"]
        start = pd.Timestamp(config.date_range.start)
        end = pd.Timestamp(config.date_range.end)
        return int(((dates >= start) & (dates <= end)).sum())

    return _cached_file("cadastre_ignitions", path, build)


def _source_families(config: Config) -> list[dict[str, Any]]:
    """One row per feature family, off the registry the dataset build is checked against."""
    from tfire.features.registry import load_registry

    return [
        {
            "category": category.name,
            "provider": category.provider or category.source,
            "period": category.period,
            "resolution": category.resolution,
            "license": category.license,
            "url": category.url,
            "temporal": category.temporal,
            "features": category.columns,
        }
        for category in load_registry().categories
        if category.role == "feature"
    ]
