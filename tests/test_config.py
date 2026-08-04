"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tfire.config import Config, load_config


def test_locked_config_values(config: Config) -> None:
    assert config.crs == "EPSG:25832"
    assert config.resolution_m == 500
    assert config.project.random_seed == 42
    assert config.fires.expected_row_count == 3187
    assert config.fires.min_valid_timestamps == 3174
    assert config.fires.near_midnight_start_hour == 22
    assert config.fires.suspicious_time_max_hour == 2


def test_date_range_is_parsed(config: Config) -> None:
    assert config.date_range.start.year == 1984
    assert config.date_range.end.year == 2024
    assert config.date_range.start < config.date_range.end


def test_relative_paths_resolve_against_project_root(config: Config) -> None:
    resolved = config.path(config.paths.fires_shapefile)
    assert resolved.is_absolute()
    assert resolved.parent.name == "pat_fires"


def test_unknown_key_is_rejected(tmp_path: Path, config: Config) -> None:
    """extra='forbid' means a typo in the YAML fails loudly rather than being ignored."""
    source = config.project_root / "config" / "config.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["resolution_metres"] = 500  # typo for resolution_m

    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(bad)
