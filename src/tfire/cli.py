"""Command-line entry points. Thin wrappers over the library code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from tfire.config import Config, load_config, setup_logging
from tfire.fires import build_positives
from tfire.grid import build_grid
from tfire.preflight import CHECKS, check_access
from tfire.sampling import build_samples

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Trentino Fire Risk pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Path to config.yaml (defaults to config/config.yaml)."),
]
ForceOption = Annotated[
    bool,
    typer.Option("--force", help="Rebuild even if the output already exists."),
]
SourceOption = Annotated[
    list[str] | None,
    typer.Option("--source", help="Source to probe; repeatable. Defaults to all."),
]


@app.callback()
def main() -> None:
    """Trentino Fire Risk pipeline.

    Registered so the app stays in sub-command mode. Without it Typer collapses a
    single-command app into its root and `build-positives` becomes unaddressable.
    """


def _start(config: Path | None) -> Config:
    cfg = load_config(config)
    setup_logging(cfg)
    logger.info(
        "Random seed: %d | CRS: %s | resolution: %dm",
        cfg.project.random_seed,
        cfg.crs,
        cfg.resolution_m,
    )
    return cfg


@app.command("build-positives")
def build_positives_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Parse the PAT fire cadastre into the positive-sample table."""
    build_positives(_start(config), force=force)


@app.command("build-grid")
def build_grid_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Build the 500 m analysis grid with its boundary and non-burnable masks."""
    build_grid(_start(config), force=force)


@app.command("build-samples")
def build_samples_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Label the positives and draw the negatives around the exclusion set."""
    build_samples(_start(config), force=force)


@app.command("check-access")
def check_access_command(config: ConfigOption = None, source: SourceOption = None) -> None:
    """Verify credentials and connectivity for the external data sources.

    Needs real credentials and network access, so it never runs in CI.
    Exits non-zero if any probe fails.
    """
    cfg = load_config(config)
    setup_logging(cfg)

    names = sorted(CHECKS) if not source else [s.lower() for s in source]
    unknown = sorted(set(names) - set(CHECKS))
    if unknown:
        raise typer.BadParameter(
            f"Unknown source(s): {', '.join(unknown)}. Choose from {sorted(CHECKS)}."
        )

    results = check_access(cfg, names)
    failed = [r.source for r in results if not r.ok]
    if failed:
        logger.error("Pre-flight failed for: %s", ", ".join(failed))
        raise typer.Exit(code=1)
    logger.info("All %d pre-flight check(s) passed", len(results))


if __name__ == "__main__":
    app()
