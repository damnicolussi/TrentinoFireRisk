"""Full-grid daily prediction: one risk map per date, from the cache or from a live forecast."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.datasets import interpolate_meteo, nearest_backbone
from tfire.features.fwi import INDEX_NAMES, compute_fwi
from tfire.features.human import calendar_features
from tfire.features.landcover import nearest_edition
from tfire.features.meteo import NOON_COLUMNS, add_lag_features, aggregate_daily, to_frame
from tfire.features.registry import Registry, load_registry
from tfire.features.vegetation import operational_window, preceding_month
from tfire.grid import GridSpec, load_grid
from tfire.models.calibration import CALIBRATOR_FILENAME, Calibrator
from tfire.models.mesogeos import MODEL_FILENAME as MESOGEOS_MODEL
from tfire.models.mesogeos import PROB_COLUMN, predict_prob
from tfire.models.trentino import (
    METRICS_FILENAME,
    Estimator,
    align_columns,
    design_matrix,
)
from tfire.raster import write_cell_bands
from tfire.sources import bias, forecast
from tfire.sources.bias import load_bias_map
from tfire.sources.era5land import read_lattice

if TYPE_CHECKING:
    from tfire.sources.era5land import Lattice
    from tfire.sources.forecast import Segment

logger = logging.getLogger(__name__)

MODEL_FILENAME: Final = "model.json"

PROBABILITY_BAND: Final = "probability"
RANK_BAND: Final = "rank"

DRIVER_COLUMNS: Final = (
    "fwi",
    "temp_mean",
    "rh_mean",
    "wind_speed_mean",
    "precip_cum30",
    "ndvi",
)


class FeatureUnavailableError(Exception):
    """No source can supply a feature the model needs for the requested date."""

    def __init__(self, day: date, features: list[str], reason: str) -> None:
        self.day = day
        self.features = features
        self.reason = reason
        super().__init__(f"{day}: cannot build {', '.join(features)}. {reason}")


@dataclass(frozen=True)
class Plan:
    """Where every dynamic block for a span comes from."""

    days: list[date]
    today: date
    meteo_from_cache: bool
    segments: list[Segment]

    @property
    def sources(self) -> list[str]:
        if self.meteo_from_cache:
            return ["cached"]
        return [f"{piece.provider}:{piece.start}..{piece.end}" for piece in self.segments]


def model_directory(config: Config) -> Path:
    return config.path(config.paths.trentino_model_dir) / config.trentino.version


def model_fingerprint(config: Config) -> str:
    """Short digest of the estimator and its calibrator, to stamp a map with what scored it."""
    directory = model_directory(config)
    digest = hashlib.sha256()
    for name in (MODEL_FILENAME, CALIBRATOR_FILENAME):
        path = directory / name
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:16]


def load_model(config: Config) -> tuple[Estimator, list[str], Calibrator]:
    """The shipped estimator, the columns it was fitted on, and its calibrator."""
    from xgboost import XGBClassifier

    directory = model_directory(config)
    for name in (MODEL_FILENAME, METRICS_FILENAME, CALIBRATOR_FILENAME):
        if not (directory / name).is_file():
            raise FileNotFoundError(
                f"{directory / name} is missing. Run `tfire train` and `tfire evaluate` first."
            )

    metrics = json.loads((directory / METRICS_FILENAME).read_text(encoding="utf-8"))
    stamp = metrics.get("config_sha256")
    if stamp != config.digest():
        # the digest covers the whole file, so one added setting is enough to move it
        logger.info("Config has changed since training: %s -> %s", stamp[:12], config.digest()[:12])

    estimator = XGBClassifier()
    estimator.load_model(directory / MODEL_FILENAME)
    calibrator = Calibrator.read(directory / CALIBRATOR_FILENAME)
    return estimator, list(metrics["columns"]), calibrator


def table_last_date(path: Path) -> date | None:
    """Latest `date` in a Parquet table, off the row-group statistics."""
    if not path.is_file():
        return None

    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    index = metadata.schema.names.index("date")
    maxima = [
        metadata.row_group(group).column(index).statistics.max
        for group in range(metadata.num_row_groups)
        if metadata.row_group(group).column(index).statistics is not None
    ]
    return pd.Timestamp(max(maxima)).date() if maxima else None


@lru_cache(maxsize=8)
def _last_date_for(path: Path, stamp: tuple[float, int] | None) -> date | None:
    return table_last_date(path)


def _stamped_last_date(path: Path) -> date | None:
    if not path.is_file():
        return None
    info = path.stat()
    return _last_date_for(path, (info.st_mtime, info.st_size))


def cached_span(config: Config) -> tuple[date, date]:
    ends = [
        _stamped_last_date(config.path(relative))
        for relative in (config.paths.meteo_out, config.paths.fwi_out)
    ]
    reached = [value for value in ends if value is not None]
    end = min(reached) if len(reached) == len(ends) else config.date_range.end
    return config.date_range.start, max(end, config.date_range.end)


def plan_days(config: Config, days: list[date], today: date | None = None) -> Plan:
    """Decide where the span's meteorology comes from, before anything is downloaded.

    The cached record wins where it reaches: it needs no network and it is the very data the
    model was fitted on. Everything else goes to Open-Meteo, which keeps a date resolvable long
    after the cached tables stop advancing.
    """
    stamp = today or date.today()
    first, last = min(days), max(days)
    cached_start, cached_end = cached_span(config)

    if first >= cached_start and last <= cached_end:
        meteo = config.path(config.paths.meteo_out)
        fwi = config.path(config.paths.fwi_out)
        if meteo.is_file() and fwi.is_file():
            return Plan(days, stamp, meteo_from_cache=True, segments=[])
        logger.info("%s or %s absent, falling back to the remote endpoints", meteo, fwi)

    horizon = stamp + timedelta(days=config.forecast.horizon_days)
    if last > horizon:
        raise FeatureUnavailableError(
            last,
            ["meteo", *INDEX_NAMES],
            f"nobody forecasts weather past {horizon}, {config.forecast.horizon_days} days out.",
        )
    if first < cached_start:
        raise FeatureUnavailableError(
            first, ["meteo", *INDEX_NAMES], f"the record starts at {cached_start}."
        )

    window_start, window_end = forecast.spinup_window(config, first, last, stamp)
    try:
        segments = forecast.plan_span(config, window_start, window_end, stamp)
    except forecast.ForecastError as error:
        raise FeatureUnavailableError(first, ["meteo", *INDEX_NAMES], str(error)) from error
    return Plan(days, stamp, meteo_from_cache=False, segments=segments)


def _backbone_daily(config: Config, plan: Plan, lattice: Lattice) -> pd.DataFrame:
    """Daily aggregates and FWI on the backbone, covering every day of the plan."""
    if plan.meteo_from_cache:
        wanted = pd.to_datetime(sorted(plan.days))
        meteo = pd.read_parquet(config.path(config.paths.meteo_out))
        fwi = pd.read_parquet(config.path(config.paths.fwi_out))
        meteo = meteo[meteo["date"].isin(wanted)]
        fwi = fwi[fwi["date"].isin(wanted)]
        return meteo.merge(fwi, on=["era5_id", "date"], how="left", validate="1:1")

    first = min(piece.start for piece in plan.segments)
    last = max(piece.end for piece in plan.segments)
    fields, _ = forecast.fetch_span(config, lattice, first, last, plan.today)

    dates, columns = aggregate_daily(fields, config)
    add_lag_features(columns, config)
    frame = to_frame(dates, columns, lattice)

    table = load_bias_map(config)
    if table is not None:
        frame = bias.correct(frame, table, config)

    fwi = compute_fwi(frame[["era5_id", "date", *NOON_COLUMNS]], lattice.cell_latitudes())
    merged = frame.merge(fwi, on=["era5_id", "date"], how="left", validate="1:1")

    # the spin-up exists so the lag columns and the FWI codes have history to stand on
    wanted = pd.to_datetime(sorted(plan.days))
    return merged[merged["date"].isin(wanted)]


def active_cells(config: Config) -> npt.NDArray[np.int64]:
    """The cells a prediction covers: the same population the negatives are drawn from."""
    _, grid = load_grid(config)
    active: npt.NDArray[np.int64] = grid.loc[grid["is_trentino"], "cell_id"].to_numpy("int64")
    return active


def static_frame(config: Config, day: date) -> pd.DataFrame:
    """One row per active cell: everything that does not move within a day."""
    _, grid = load_grid(config)
    frame = grid.loc[grid["is_trentino"], ["cell_id"]].reset_index(drop=True)

    for relative in (
        config.paths.topography_out,
        config.paths.geography_out,
        config.paths.human_out,
    ):
        block = pd.read_parquet(config.path(relative))
        frame = frame.merge(block, on="cell_id", how="left", validate="1:1")

    stamp = pd.Series([pd.Timestamp(day)])
    edition = int(nearest_edition(stamp, config.corine.editions).iloc[0])
    landcover = pd.read_parquet(config.path(config.paths.landcover_out))
    frame = frame.merge(
        landcover[landcover["clc_edition"] == edition].drop(columns="clc_edition"),
        on="cell_id",
        how="left",
        validate="1:1",
    )

    year = int(nearest_edition(stamp, config.human.worldpop_years).iloc[0])
    population = pd.read_parquet(config.path(config.paths.human_population_out))
    frame = frame.merge(
        population[population["worldpop_year"] == year].drop(columns="worldpop_year"),
        on="cell_id",
        how="left",
        validate="1:1",
    )

    frame["clc_edition"] = np.int16(edition)
    frame["worldpop_year"] = np.int16(year)
    return frame


def assemble_day(
    config: Config,
    day: date,
    statics: pd.DataFrame,
    backbone: pd.DataFrame,
    weights: pd.DataFrame,
    vegetation: pd.DataFrame,
) -> pd.DataFrame:
    """The full-grid feature table for one date, in the shape the training table had."""
    frame = statics.copy()
    frame["date"] = pd.Timestamp(day)

    daily = backbone[backbone["date"] == pd.Timestamp(day)]
    if daily.empty:
        raise FeatureUnavailableError(day, ["meteo"], "the backbone carries no row for this date.")

    linear = daily.drop(columns=list(INDEX_NAMES))
    frame = frame.merge(
        interpolate_meteo(frame, weights, linear, key="cell_id"),
        on="cell_id",
        how="left",
        validate="1:1",
    )

    codes = daily[["era5_id", *INDEX_NAMES]]
    frame = frame.merge(nearest_backbone(weights), on="cell_id", how="left", validate="m:1")
    frame = frame.merge(codes, on="era5_id", how="left", validate="m:1").drop(columns="era5_id")

    frame = frame.merge(vegetation, on="cell_id", how="left", validate="1:1")
    frame["veg_composite_date"] = pd.Timestamp(preceding_month(day))

    calendar = calendar_features(pd.DatetimeIndex([pd.Timestamp(day)]))
    frame = frame.merge(calendar, on="date", how="left", validate="m:1")

    if (config.path(config.paths.mesogeos_model_dir) / MESOGEOS_MODEL).is_file():
        frame[PROB_COLUMN] = predict_prob(frame, config)
    return frame


def check_contract(frame: pd.DataFrame, registry: Registry, day: date) -> None:
    """Every feature the registry declares has to be on the table before anything is scored."""
    declared = {spec.name for spec in registry.features if not spec.optional}
    missing = sorted(declared - set(frame.columns))
    if missing:
        raise FeatureUnavailableError(
            day, missing, "the feature availability contract is not met for this date."
        )


def score(
    frame: pd.DataFrame,
    registry: Registry,
    estimator: Estimator,
    columns: list[str],
    calibrator: Calibrator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Population-rate probability per row, and its rank within the day."""
    features, _, _ = design_matrix(frame.assign(is_fire=False), registry)
    aligned = align_columns(features, columns)

    raw = estimator.predict_proba(aligned)[:, 1]
    probability = calibrator.to_population_rate(raw)
    rank = np.asarray(pd.Series(probability).rank(pct=True), dtype="float64")
    return probability, rank


def _outputs(config: Config, day: date) -> dict[str, Path]:
    directory = config.path(config.paths.risk_dir)
    return {
        "raster": directory / f"risk_{day:%Y-%m-%d}.tif",
        "table": directory / f"risk_{day:%Y-%m-%d}.parquet",
        "meta": directory / f"risk_{day:%Y-%m-%d}.json",
    }


def write_day(
    config: Config,
    day: date,
    spec: GridSpec,
    frame: pd.DataFrame,
    probability: npt.NDArray[np.float64],
    rank: npt.NDArray[np.float64],
    provenance: dict[str, Any],
) -> dict[str, Path]:
    paths = _outputs(config, day)

    table = pd.DataFrame(
        {
            "cell_id": frame["cell_id"].to_numpy("int32"),
            PROBABILITY_BAND: probability.astype("float32"),
            RANK_BAND: rank.astype("float32"),
            **{name: frame[name].to_numpy("float32") for name in DRIVER_COLUMNS},
        }
    )
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths["table"], index=False)

    bands = {}
    for name, values in ((PROBABILITY_BAND, probability), (RANK_BAND, rank)):
        full = np.full(spec.n_cells, np.nan)
        full[table["cell_id"].to_numpy()] = values
        bands[name] = full
    write_cell_bands(spec, paths["raster"], bands)

    paths["meta"].write_text(
        json.dumps(provenance, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    return paths


def _provenance(
    config: Config, day: date, plan: Plan, frame: pd.DataFrame, probability: npt.NDArray[np.float64]
) -> dict[str, Any]:
    empty = {
        name: int(frame[name].isna().sum()) for name in frame.columns if frame[name].isna().any()
    }
    dated = frame["age_days"].notna()
    return {
        "date": day,
        "model_version": config.trentino.version,
        "model_fingerprint": model_fingerprint(config),
        "sources": plan.sources,
        "bias_map": None if plan.meteo_from_cache else bias.bias_fingerprint(config),
        "cells": len(frame),
        "vegetation": {
            "composite": frame["veg_composite_date"].iloc[0],
            # null rather than NaN: a bare NaN is not JSON, and whatever reads this next is a
            # parser, not Python
            "mean_age_days": float(frame.loc[dated, "age_days"].mean()) if dated.any() else None,
            "cells_with_ndvi": int(frame["ndvi"].notna().sum()),
        },
        "probability": {
            "mean": float(np.mean(probability)),
            "max": float(np.max(probability)),
        },
        "missing_by_feature": empty,
        "config_sha256": config.digest(),
    }


class GridScorer:
    """Scores whole days on the active grid, reusing the blocks that do not move between them."""

    def __init__(self, config: Config, days: list[date], today: date | None = None) -> None:
        self.config = config
        self.plan = plan_days(config, days, today)
        self.registry = load_registry()
        self.estimator, self.columns, self.calibrator = load_model(config)
        self.spec, _ = load_grid(config)

        lattice = read_lattice(config)
        self._weights = pd.read_parquet(config.path(config.paths.era5_weights_out))
        self._backbone = _backbone_daily(config, self.plan, lattice)

        # both blocks are step functions of the date, so a span rarely needs more than one each
        self._statics: dict[tuple[int, int], pd.DataFrame] = {}
        self._vegetation: dict[date, pd.DataFrame] = {}

    def day(
        self, day: date
    ) -> tuple[pd.DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        stamp = pd.Series([pd.Timestamp(day)])
        key = (
            int(nearest_edition(stamp, self.config.corine.editions).iloc[0]),
            int(nearest_edition(stamp, self.config.human.worldpop_years).iloc[0]),
        )
        if key not in self._statics:
            self._statics[key] = static_frame(self.config, day)

        composite = preceding_month(day)
        if composite not in self._vegetation:
            self._vegetation[composite] = operational_window(self.config, composite)

        frame = assemble_day(
            self.config,
            day,
            self._statics[key],
            self._backbone,
            self._weights,
            self._vegetation[composite],
        )
        check_contract(frame, self.registry, day)
        probability, rank = score(
            frame, self.registry, self.estimator, self.columns, self.calibrator
        )
        return frame, probability, rank


def predict_days(
    config: Config,
    start: date,
    days: int = 1,
    force: bool = False,
    today: date | None = None,
) -> list[date]:
    """Write a risk map for each of `days` dates from `start`, one day at a time."""
    wanted = [start + timedelta(days=offset) for offset in range(days)]
    pending = [day for day in wanted if force or not _outputs(config, day)["raster"].is_file()]
    if not pending:
        logger.info("All %d day(s) already written (use --force to rebuild)", len(wanted))
        return []

    scorer = GridScorer(config, pending, today)
    logger.info(
        "Predicting %d day(s), %s to %s, from %s",
        len(pending),
        min(pending),
        max(pending),
        ", ".join(scorer.plan.sources),
    )

    written = []
    for day in pending:
        frame, probability, rank = scorer.day(day)
        paths = write_day(
            config,
            day,
            scorer.spec,
            frame,
            probability,
            rank,
            _provenance(config, day, scorer.plan, frame, probability),
        )
        logger.info(
            "%s | mean %.2e, max %.2e over %d cell(s) -> %s",
            day,
            float(np.mean(probability)),
            float(np.max(probability)),
            len(frame),
            paths["raster"].name,
        )
        written.append(day)

    return written
