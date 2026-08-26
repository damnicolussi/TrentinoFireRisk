"""Shared fixtures. Tests never touch the real shapefile."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfire.config import Config, load_config

MODEL_ARTIFACTS = ("model.json", "metrics.json", "calibrator.json")


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


def requires_built(config: Config, *relative: Path) -> None:
    """Skip when an artifact the pipeline produces is absent."""
    for path in (config.path(item) for item in relative):
        if not path.exists():
            pytest.skip(f"not built in this checkout: {path}")


def requires_model(config: Config, *extra: str) -> None:
    """Skip unless the shipped version carries every artifact it is loaded from."""
    directory = Path(config.paths.trentino_model_dir) / config.trentino.version
    requires_built(config, *(directory / name for name in (*MODEL_ARTIFACTS, *extra)))
