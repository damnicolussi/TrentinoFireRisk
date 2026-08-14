"""Typed configuration loaded from `config/config.yaml`.

Every constant used by the pipeline lives in the YAML file.
Models use `extra="forbid"` so a typo in the YAML fails loudly at load time
instead of silently falling back to a default.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(extra="forbid")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until a directory containing `pyproject.toml` is found.

    Lets the CLI be invoked from any working directory while keeping every path in
    the config file relative to the repository root.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found in any parent of {current}")


class ProjectConfig(BaseModel):
    model_config = _STRICT

    random_seed: int


class DateRangeConfig(BaseModel):
    model_config = _STRICT

    start: date
    end: date


class PathsConfig(BaseModel):
    model_config = _STRICT

    raw: Path
    interim: Path
    processed: Path
    fires_shapefile: Path
    fires_out: Path
    fire_polygons_out: Path
    pat_boundary: Path
    pat_waterbodies: Path
    corine_raw: Path
    corine_out: Path
    dem_raster: Path
    grid_out: Path
    grid_spec_out: Path
    samples_out: Path
    exclusions_out: Path
    topography_out: Path
    geography_out: Path
    landcover_out: Path
    era5_raw: Path
    era5_weights_out: Path
    meteo_out: Path
    fwi_out: Path
    landsat_raw: Path
    vegetation_out: Path
    vegetation_climatology_out: Path
    vegetation_missingness_out: Path
    pat_natura2000: Path
    osm_cache: Path
    worldpop_raster: Path
    human_out: Path
    human_population_out: Path
    mesogeos_raw: Path
    mesogeos_model_dir: Path
    dataset_out: Path
    quality_report_out: Path


class CorineConfig(BaseModel):
    model_config = _STRICT

    margin_m: float = Field(gt=0)
    editions: dict[int, Path]


class GridConfig(BaseModel):
    model_config = _STRICT

    expected_active_cells: int = Field(gt=0)
    non_burnable_clc_codes: list[int]
    non_burnable_threshold: float = Field(gt=0, le=1)
    non_burnable_clc_edition: int


class BBoxWGS84(BaseModel):
    """Lat/lon bounding box used to crop external API requests server-side."""

    model_config = _STRICT

    north: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def check_ordering(self) -> BBoxWGS84:
        if self.north <= self.south:
            raise ValueError(f"north ({self.north}) must exceed south ({self.south})")
        if self.east <= self.west:
            raise ValueError(f"east ({self.east}) must exceed west ({self.west})")
        return self

    def as_bounds(self) -> list[float]:
        """Earth Engine ordering: xmin, ymin, xmax, ymax."""
        return [self.west, self.south, self.east, self.north]


class SourcesConfig(BaseModel):
    model_config = _STRICT

    bbox_wgs84: BBoxWGS84
    gee_project: str
    dem_collection: str
    dem_scale_m: int = Field(gt=0)


class FiresConfig(BaseModel):
    model_config = _STRICT

    expected_row_count: int
    min_valid_timestamps: int
    near_midnight_start_hour: int = Field(ge=0, le=23)
    near_midnight_end_hour: int = Field(ge=0, le=23)
    suspicious_time_max_hour: int = Field(ge=0, le=23)
    expected_normalized_rows: int


class TopographyConfig(BaseModel):
    model_config = _STRICT

    min_elevation_m: float
    max_elevation_m: float

    @model_validator(mode="after")
    def check_ordering(self) -> TopographyConfig:
        if self.max_elevation_m <= self.min_elevation_m:
            raise ValueError(
                f"max_elevation_m ({self.max_elevation_m}) must exceed "
                f"min_elevation_m ({self.min_elevation_m})"
            )
        return self


class GeographyConfig(BaseModel):
    model_config = _STRICT

    waterbody_threshold: float = Field(gt=0, le=1)


class MeteoConfig(BaseModel):
    model_config = _STRICT

    variables: list[str] = Field(min_length=1)
    utc_offset_hours: int = Field(ge=-12, le=14)
    spinup_years: int = Field(ge=0)
    gee_workers: int = Field(gt=0)
    gee_window_hours: int = Field(gt=0)
    precip_windows: list[int] = Field(min_length=1)
    temp_window_days: int = Field(gt=0)
    rh_window_days: int = Field(gt=0)
    min_temp_c: float
    max_temp_c: float
    min_pressure_hpa: float = Field(gt=0)
    max_pressure_hpa: float = Field(gt=0)
    max_precip_mm: float = Field(gt=0)

    @model_validator(mode="after")
    def check_ordering(self) -> MeteoConfig:
        if self.max_temp_c <= self.min_temp_c:
            raise ValueError(
                f"max_temp_c ({self.max_temp_c}) must exceed min_temp_c ({self.min_temp_c})"
            )
        if self.max_pressure_hpa <= self.min_pressure_hpa:
            raise ValueError(
                f"max_pressure_hpa ({self.max_pressure_hpa}) must exceed "
                f"min_pressure_hpa ({self.min_pressure_hpa})"
            )
        if any(window < 1 for window in self.precip_windows):
            raise ValueError(f"precip_windows must be positive, got {self.precip_windows}")
        return self

    @property
    def longest_window_days(self) -> int:
        """Days of history the lag features need before the first emitted day."""
        return max(*self.precip_windows, self.temp_window_days, self.rh_window_days)


class FWIConfig(BaseModel):
    model_config = _STRICT

    max_ffmc: float = Field(gt=0)
    max_dmc: float = Field(gt=0)
    max_dc: float = Field(gt=0)
    max_isi: float = Field(gt=0)
    max_bui: float = Field(gt=0)
    max_fwi: float = Field(gt=0)


class VegetationConfig(BaseModel):
    model_config = _STRICT

    collections: list[str] = Field(min_length=1)
    chunk_months: int = Field(gt=0, le=12)
    mask_snow: bool
    min_valid_fraction: float = Field(ge=0, le=1)
    max_fill_days: int = Field(gt=0)
    climatology_min_years: int = Field(gt=0)
    no_observation_strategy: Literal["nan", "climatology"]
    min_index: float
    max_index: float
    min_lst_c: float
    max_lst_c: float

    @model_validator(mode="after")
    def check_ordering(self) -> VegetationConfig:
        if self.max_index <= self.min_index:
            raise ValueError(
                f"max_index ({self.max_index}) must exceed min_index ({self.min_index})"
            )
        if self.max_lst_c <= self.min_lst_c:
            raise ValueError(
                f"max_lst_c ({self.max_lst_c}) must exceed min_lst_c ({self.min_lst_c})"
            )
        if 12 % self.chunk_months:
            raise ValueError(f"chunk_months must divide 12, got {self.chunk_months}")
        return self


class HumanConfig(BaseModel):
    model_config = _STRICT

    osm_distance_subpoints_per_side: int = Field(gt=0)
    osm_overpass_timeout_s: int = Field(gt=0)
    natura2000_threshold: float = Field(gt=0, le=1)
    worldpop_collection: str
    worldpop_years: list[int] = Field(min_length=1)
    max_pop_density_km2: float = Field(gt=0)

    @model_validator(mode="after")
    def check_worldpop_years(self) -> HumanConfig:
        years = self.worldpop_years
        if sorted(set(years)) != years:
            raise ValueError(
                f"worldpop_years must be strictly increasing with no duplicates: {years}"
            )
        if any(not (2000 <= year <= 2020) for year in years):
            raise ValueError(f"WorldPop/GP/100m/pop covers 2000-2020, got {years}")
        return self


class SamplingConfig(BaseModel):
    model_config = _STRICT

    negative_ratio: int = Field(gt=0)
    buffer_cells: int = Field(ge=0)
    buffer_days: int = Field(ge=0)
    hard_negative_fraction: float = Field(ge=0, le=1)
    expected_positives: int = Field(gt=0)


class MesogeosConfig(BaseModel):
    model_config = _STRICT

    track_length_days: int = Field(gt=0)
    holdout_years: list[int] = Field(min_length=1)
    max_depth: int = Field(gt=0)
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=1)
    early_stopping_rounds: int = Field(gt=0)

    @property
    def target_time_idx(self) -> int:
        """Index of the day the label refers to, the last of the track."""
        return self.track_length_days - 1


class DatasetConfig(BaseModel):
    model_config = _STRICT

    expected_rows: int = Field(gt=0)
    expected_features: int = Field(gt=0)


class LoggingConfig(BaseModel):
    model_config = _STRICT

    level: str


class Config(BaseModel):
    model_config = _STRICT

    project: ProjectConfig
    crs: str
    resolution_m: int
    date_range: DateRangeConfig
    paths: PathsConfig
    sources: SourcesConfig
    corine: CorineConfig
    grid: GridConfig
    fires: FiresConfig
    topography: TopographyConfig
    geography: GeographyConfig
    meteo: MeteoConfig
    fwi: FWIConfig
    vegetation: VegetationConfig
    human: HumanConfig
    sampling: SamplingConfig
    mesogeos: MesogeosConfig
    dataset: DatasetConfig
    logging: LoggingConfig

    project_root: Path

    def path(self, relative: Path) -> Path:
        """Resolve a config-declared path against the project root."""
        return relative if relative.is_absolute() else self.project_root / relative


def load_config(path: Path | None = None) -> Config:
    """Load and validate the master config.

    Defaults to `<project_root>/config/config.yaml`.
    """
    root = find_project_root()
    config_path = path if path is not None else root / "config" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return Config(**raw, project_root=root)


def setup_logging(config: Config) -> None:
    """Configure the root logger. The pipeline never uses `print`."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
