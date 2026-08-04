"""Shared fixtures. Tests never touch the real shapefile."""

from __future__ import annotations

import pytest

from tfire.config import Config, load_config


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()
