"""The subset of the data tree a serving container needs, assembled and measured."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from tfire.config import Config
from tfire.models.calibration import CALIBRATOR_FILENAME
from tfire.models.danger import DANGER_FILENAME
from tfire.models.trentino import METRICS_FILENAME

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: Final = "manifest.json"


@dataclass(frozen=True)
class Entry:
    relative: str
    bytes: int
    required: bool


def runtime_paths(config: Config) -> dict[Path, bool]:
    """Every path a prediction and the API read, mapped to whether it has to be there.

    Three entries come out of `data/raw` because inference genuinely reads them: the Landsat
    composites, the provincial boundary and the fire inventory. The rest of the 4.1 GB is build
    time only. `tests/test_api.py` records the paths a prediction and the public endpoints
    resolve and checks them against this list, since a missing entry produces a container that
    works on the build host and fails, or quietly loses a feature block, in production.
    """
    paths = config.paths
    required = [
        paths.grid_out,
        paths.grid_spec_out,
        paths.topography_out,
        paths.geography_out,
        paths.landcover_out,
        paths.human_out,
        paths.human_population_out,
        paths.era5_weights_out,
        paths.meteo_out,
        paths.fwi_out,
        # the historical-fires layer and the boundary outline are part of the public map, not
        # build artifacts
        paths.fires_out,
        paths.pat_boundary,
        # inference builds its vegetation block from the composites themselves, not from
        # `vegetation_out`: without them every served map carries an empty one
        paths.landsat_raw,
    ]
    optional = [
        # only the dates past the cached backbone go through it, and a deployment that never
        # serves those can do without
        paths.bias_map_out,
        paths.fire_polygons_out,
        # 270 MB the training build reads and inference does not: it takes its vegetation from the
        # composites. Shipped only so a mounted build tree stays complete.
        paths.vegetation_out,
        paths.dataset_out,
        paths.quality_report_out,
        paths.mesogeos_model_dir,
        paths.report_dir,
    ]

    model_dir = Path(paths.trentino_model_dir) / config.trentino.version
    required += [model_dir / name for name in ("model.json", METRICS_FILENAME, CALIBRATOR_FILENAME)]
    optional += [model_dir / DANGER_FILENAME, model_dir / "evaluation.json"]
    # the admin drawer offers every version it finds beside the active one, so shipping only the
    # active one leaves the rollback control pointing at a directory that is not in the container
    optional += _other_versions(config)

    return {**{path: True for path in required}, **{path: False for path in optional}}


def _other_versions(config: Config) -> list[Path]:
    root = config.path(config.paths.trentino_model_dir)
    if not root.is_dir():
        return []
    relative = Path(config.paths.trentino_model_dir)
    return [
        relative / item.name
        for item in sorted(root.iterdir())
        if item.is_dir() and item.name != config.trentino.version
    ]


def members(path: Path) -> list[Path]:
    """The files one entry stands for.

    A shapefile is a set: the geometry in `.shp`, the index in `.shx`, the attributes in `.dbf`,
    the projection in `.prj`. Copying the path the config names moves one of them and leaves a
    file that no reader can open.
    """
    if path.is_dir():
        return [item for item in sorted(path.rglob("*")) if item.is_file()]
    if path.suffix == ".shp":
        return sorted(
            sibling for sibling in path.parent.glob(f"{path.stem}.*") if sibling.is_file()
        )
    return [path]


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in members(path))


def build_manifest(config: Config) -> dict[str, Any]:
    entries = []
    missing = []
    for relative, required in runtime_paths(config).items():
        absolute = config.path(relative)
        if not absolute.exists():
            if required:
                missing.append(str(relative))
            continue
        entries.append(Entry(str(relative), _size(absolute), required))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required runtime file(s) absent, the pipeline has not been run to "
            f"the end: {', '.join(missing)}"
        )

    return {
        "generated": date.today().isoformat(),
        "model_version": config.trentino.version,
        "config_sha256": config.digest(),
        "total_bytes": sum(entry.bytes for entry in entries),
        "entries": [entry.__dict__ for entry in sorted(entries, key=lambda e: -e.bytes)],
    }


def package_runtime(config: Config, out: Path, force: bool = False) -> Path:
    """Copy the runtime subset into `out`, preserving the layout the config expects."""
    manifest = build_manifest(config)
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists, pass --force to replace it")
    if out.exists():
        shutil.rmtree(out)

    for entry in manifest["entries"]:
        source = config.path(Path(entry["relative"]))
        target = out / entry["relative"]
        if source.is_dir():
            shutil.copytree(source, target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        for member in members(source):
            shutil.copy2(member, target.parent / member.name)

    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config.path(config.paths.runtime_manifest_out).parent.mkdir(parents=True, exist_ok=True)
    config.path(config.paths.runtime_manifest_out).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info(
        "Packaged %d item(s), %.0f MB, into %s",
        len(manifest["entries"]),
        manifest["total_bytes"] / 1e6,
        out,
    )
    for entry in manifest["entries"][:5]:
        logger.info("  %8.0f MB  %s", entry["bytes"] / 1e6, entry["relative"])
    return out
