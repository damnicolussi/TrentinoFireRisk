"""Feature metadata loaded from `config/features.yaml`, and the check that enforces it."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from tfire.config import Config, find_project_root

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(extra="forbid")
_REFERENCE = "config:"

Referable = float | int | bool | str | None


def resolve(value: Referable, config: Config) -> Referable:
    """Follow a `config:<dotted.path>` reference into the master config, or pass the value on."""
    if not isinstance(value, str) or not value.startswith(_REFERENCE):
        return value
    node: object = config
    for part in value.removeprefix(_REFERENCE).split("."):
        node = getattr(node, part)
    return cast(Referable, node)


class FeatureSpec(BaseModel):
    model_config = _STRICT

    name: str
    category: str
    source: str
    dtype: str
    role: Literal["key", "label", "feature", "diagnostic"] = "feature"
    temporal: Literal["static", "dynamic"] | None = None
    forecastable: bool | None = None
    nullable: bool = False
    optional: bool = False
    min: float | str | None = None
    max: float | str | None = None

    def bounds(self, config: Config) -> tuple[float | None, float | None]:
        lower, upper = resolve(self.min, config), resolve(self.max, config)
        return (
            None if lower is None else float(lower),
            None if upper is None else float(upper),
        )


class Registry:
    """The declared columns of the training table, in file order."""

    def __init__(self, specs: Sequence[FeatureSpec]) -> None:
        self.specs = tuple(specs)

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return tuple(spec for spec in self.specs if spec.role == "feature")

    def present(self, frame: pd.DataFrame) -> tuple[FeatureSpec, ...]:
        """The declared features the table actually carries, optional ones included."""
        return tuple(spec for spec in self.features if spec.name in frame.columns)


def _flatten(raw: dict[str, Any]) -> Iterator[FeatureSpec]:
    for category, block in raw["categories"].items():
        defaults = dict(block)
        columns = defaults.pop("columns")
        for name, overrides in columns.items():
            merged = {**defaults, **(overrides or {})}
            yield FeatureSpec(name=name, category=category, **merged)


def load_registry(path: Path | None = None) -> Registry:
    """Defaults to `<project_root>/config/features.yaml`."""
    registry_path = path if path is not None else find_project_root() / "config" / "features.yaml"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Feature registry not found: {registry_path}")

    with registry_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    specs = list(_flatten(raw))
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Feature declared more than once in {registry_path}: {duplicates}")

    missing = [spec.name for spec in specs if spec.role == "feature" and spec.temporal is None]
    if missing:
        raise ValueError(f"Features without a `temporal` in {registry_path}: {missing}")

    return Registry(specs)


def validate_frame(frame: pd.DataFrame, registry: Registry, config: Config) -> None:
    """Check the assembled table against the registry, reporting every problem before raising."""
    declared = {spec.name: spec for spec in registry.specs}

    problems: list[str] = []
    for column in frame.columns:
        if column not in declared:
            problems.append(f"{column}: present in the table, absent from the registry")
    for name, spec in declared.items():
        if name not in frame.columns and not spec.optional:
            problems.append(f"{name}: declared in the registry, absent from the table")

    for name, spec in declared.items():
        if name not in frame.columns:
            continue
        values = frame[name]

        actual = str(values.dtype)
        if actual != spec.dtype:
            problems.append(f"{name}: dtype is {actual}, registry says {spec.dtype}")

        nulls = int(values.isna().sum())
        if nulls and not spec.nullable:
            problems.append(f"{name}: {nulls} null value(s) in a non-nullable column")

        lower, upper = spec.bounds(config)
        if lower is None and upper is None:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        below = 0 if lower is None else int((numeric < lower).sum())
        above = 0 if upper is None else int((numeric > upper).sum())
        if below or above:
            problems.append(
                f"{name}: {below} below {lower} and {above} above {upper} "
                f"(observed {numeric.min()} to {numeric.max()})"
            )

    for problem in problems:
        logger.error("Registry violation | %s", problem)
    if problems:
        raise ValueError(f"{len(problems)} registry violation(s), see the log above")

    logger.info(
        "Registry check passed: %d column(s), %d of them features",
        len(frame.columns),
        len(registry.present(frame)),
    )
