"""Command-line entry points. Thin wrappers over the library code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from tfire.config import Config, load_config, setup_logging
from tfire.datasets import build_dataset
from tfire.features import EXTRACTORS
from tfire.fires import build_positives
from tfire.grid import build_grid
from tfire.models.mesogeos import train_mesogeos
from tfire.models.trentino import SPECS, train_trentino
from tfire.preflight import CHECKS, check_access
from tfire.sampling import build_samples
from tfire.sources.era5land import fetch_era5, fetch_years
from tfire.sources.landsat import fetch_landsat

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
CategoryOption = Annotated[
    list[str] | None,
    typer.Option("--category", help="Feature category to extract; repeatable. Defaults to all."),
]
ModelOption = Annotated[
    list[str] | None,
    typer.Option("--model", help="Model to train; repeatable. Defaults to all."),
]
YearOption = Annotated[
    list[int] | None,
    typer.Option("--year", help="Year to download; repeatable. Defaults to the whole record."),
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


@app.command("extract-features")
def extract_features_command(
    config: ConfigOption = None, category: CategoryOption = None, force: ForceOption = False
) -> None:
    """Extract one or more feature categories onto the grid."""
    selected = set(EXTRACTORS) if not category else {c.lower() for c in category}
    unknown = sorted(selected - set(EXTRACTORS))
    if unknown:
        raise typer.BadParameter(
            f"Unknown categor{'y' if len(unknown) == 1 else 'ies'}: {', '.join(unknown)}. "
            f"Choose from {sorted(EXTRACTORS)}."
        )

    # registry order, not the order asked for: fwi reads what meteo writes
    names = [name for name in EXTRACTORS if name in selected]
    cfg = _start(config)
    for name in names:
        logger.info("Extracting %s", name)
        EXTRACTORS[name](cfg, force)


@app.command("train-mesogeos")
def train_mesogeos_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Train the stacking base model on the Mesogeos tracks."""
    train_mesogeos(_start(config), force=force)


@app.command("train")
def train_command(
    config: ConfigOption = None, model: ModelOption = None, force: ForceOption = False
) -> None:
    """Train the Trentino model and its baselines on the assembled table."""
    names = None if not model else [name.lower() for name in model]
    if names:
        unknown = sorted(set(names) - set(SPECS))
        if unknown:
            raise typer.BadParameter(
                f"Unknown model(s): {', '.join(unknown)}. Choose from {sorted(SPECS)}."
            )
        names = [name for name in SPECS if name in set(names)]

    train_trentino(_start(config), force=force, selected=names)


@app.command("build-dataset")
def build_dataset_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Join the samples against every feature block into the training table."""
    build_dataset(_start(config), force=force)


@app.command("fetch-era5")
def fetch_era5_command(
    config: ConfigOption = None, year: YearOption = None, force: ForceOption = False
) -> None:
    """Download the ERA5-Land hourly fields from Earth Engine into the raw cache."""
    cfg = _start(config)
    available = list(fetch_years(cfg))
    years = available if not year else sorted(set(year))

    outside = [value for value in years if value not in available]
    if outside:
        raise typer.BadParameter(
            f"Year(s) outside the record: {outside}. Choose from {available[0]} to {available[-1]}."
        )

    fetch_era5(cfg, years, force=force)


@app.command("fetch-landsat")
def fetch_landsat_command(
    config: ConfigOption = None, year: YearOption = None, force: ForceOption = False
) -> None:
    """Download the monthly Landsat composites from Earth Engine into the raw cache."""
    cfg = _start(config)
    available = list(range(cfg.date_range.start.year, cfg.date_range.end.year + 1))
    years = available if not year else sorted(set(year))

    outside = [value for value in years if value not in available]
    if outside:
        raise typer.BadParameter(
            f"Year(s) outside the record: {outside}. Choose from {available[0]} to {available[-1]}."
        )

    fetch_landsat(cfg, years, force=force)


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
