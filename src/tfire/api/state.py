"""Which model version the service is serving, kept outside the tracked config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from tfire.config import Config
from tfire.models.calibration import CALIBRATOR_FILENAME
from tfire.models.trentino import METRICS_FILENAME

logger = logging.getLogger(__name__)

MODEL_FILENAME: Final = "model.json"
_REQUIRED: Final = (MODEL_FILENAME, METRICS_FILENAME, CALIBRATOR_FILENAME)


def available_versions(config: Config) -> list[str]:
    """Version directories carrying a complete, loadable set of artifacts."""
    root = config.path(config.paths.trentino_model_dir)
    if not root.is_dir():
        return []
    return sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and all((item / name).is_file() for name in _REQUIRED)
    )


def _state_path(config: Config) -> Path:
    return config.path(config.paths.app_state_out)


def read_state(config: Config) -> dict[str, str]:
    path = _state_path(config)
    if not path.is_file():
        return {}
    state: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return state


def active_version(config: Config) -> str:
    """The operator's choice if it is still installable, otherwise the config's own."""
    chosen = read_state(config).get("model_version")
    if chosen and chosen in available_versions(config):
        return chosen
    if chosen:
        logger.warning("Selected model version %s is no longer available, ignoring it", chosen)
    return config.trentino.version


def select_version(config: Config, version: str) -> str:
    if version not in available_versions(config):
        raise ValueError(f"Unknown model version {version!r}, have {available_versions(config)}")

    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**read_state(config), "model_version": version}, indent=2))
    logger.info("Active model version is now %s", version)
    return version


def active_config(config: Config) -> Config:
    """The config every request should work against, with the operator's version applied."""
    version = active_version(config)
    if version == config.trentino.version:
        return config
    resolved = config.model_copy(deep=True)
    resolved.trentino.version = version
    return resolved
