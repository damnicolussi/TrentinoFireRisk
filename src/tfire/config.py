"""Typed configuration loaded from `config/config.yaml`.

Every constant used by the pipeline lives in the YAML file.
Models use `extra="forbid"` so a typo in the YAML fails loudly at load time
instead of silently falling back to a default.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

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

    def as_cds_area(self) -> list[float]:
        """CDS `area` ordering: north, west, south, east."""
        return [self.north, self.west, self.south, self.east]

    def as_bounds(self) -> list[float]:
        """Standard GIS/GEE ordering: xmin, ymin, xmax, ymax."""
        return [self.west, self.south, self.east, self.north]


class SourcesConfig(BaseModel):
    model_config = _STRICT

    bbox_wgs84: BBoxWGS84
    gee_project: str


class FiresConfig(BaseModel):
    model_config = _STRICT

    expected_row_count: int
    min_valid_timestamps: int
    near_midnight_start_hour: int = Field(ge=0, le=23)
    near_midnight_end_hour: int = Field(ge=0, le=23)
    suspicious_time_max_hour: int = Field(ge=0, le=23)
    expected_normalized_rows: int


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
    fires: FiresConfig
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
