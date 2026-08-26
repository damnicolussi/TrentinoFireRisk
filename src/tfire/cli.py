"""Command-line entry points. Thin wrappers over the library code."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from tfire.config import Config, load_config, setup_logging
from tfire.datasets import build_dataset
from tfire.features import EXTRACTORS
from tfire.features.vegetation import refresh_landsat
from tfire.fires import build_positives
from tfire.grid import build_grid
from tfire.inference import FeatureUnavailableError, model_directory, predict_days
from tfire.models.danger import build_danger_classes
from tfire.models.evaluate import evaluate_trentino
from tfire.models.events import EVENTS_FILENAME, verify_events
from tfire.models.mesogeos import train_mesogeos
from tfire.models.trentino import SPECS, train_trentino
from tfire.packaging import package_runtime
from tfire.preflight import CHECKS, check_access
from tfire.sampling import build_samples
from tfire.sensitivity import VARIANTS
from tfire.sources.bias import fit_bias_map
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
VariantOption = Annotated[
    list[str] | None,
    typer.Option(
        "--sensitivity",
        help="Sensitivity variant to run; repeatable, `all` for every one. Defaults to none.",
    ),
]
YearOption = Annotated[
    list[int] | None,
    typer.Option("--year", help="Year to download; repeatable. Defaults to the whole record."),
]
DateOption = Annotated[
    datetime | None,
    typer.Option("--date", formats=["%Y-%m-%d"], help="Date to predict. Defaults to today."),
]
DaysOption = Annotated[
    int,
    typer.Option("--days", min=1, help="Consecutive days to predict, starting at --date."),
]
RefreshOption = Annotated[
    bool,
    typer.Option(
        "--refresh-vegetation",
        help="Download any Landsat composite the requested dates need but the cache lacks.",
    ),
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


@app.command("evaluate")
def evaluate_command(
    config: ConfigOption = None,
    sensitivity: VariantOption = None,
    force: ForceOption = False,
) -> None:
    """Score, attribute and calibrate the trained model, and write the evaluation figures."""
    if sensitivity:
        unknown = sorted(set(sensitivity) - set(VARIANTS) - {"all"})
        if unknown:
            raise typer.BadParameter(
                f"Unknown variant(s): {', '.join(unknown)}. Choose from {sorted(VARIANTS)} or all."
            )

    evaluate_trentino(_start(config), force=force, variants=sensitivity)


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


@app.command("predict")
def predict_command(
    config: ConfigOption = None,
    day: DateOption = None,
    days: DaysOption = 1,
    refresh_vegetation: RefreshOption = False,
    force: ForceOption = False,
) -> None:
    """Write the risk map for a date, and optionally for the days after it."""
    cfg = _start(config)
    start = day.date() if day else date.today()

    if refresh_vegetation:
        refresh_landsat(cfg, start + timedelta(days=days - 1))

    try:
        predict_days(cfg, start, days=days, force=force)
    except FeatureUnavailableError as error:
        logger.error("%s", error)
        raise typer.Exit(code=1) from error


@app.command("danger-classes")
def danger_classes_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Derive the absolute danger-class breaks the map legend uses."""
    build_danger_classes(_start(config), force=force)


@app.command("fit-bias-map")
def fit_bias_map_command(config: ConfigOption = None, force: ForceOption = False) -> None:
    """Fit the quantile map that puts remotely served fields back on the backbone."""
    fit_bias_map(_start(config), force=force)


@app.command("verify-events")
def verify_events_command(config: ConfigOption = None) -> None:
    """Score every day that carries a recorded ignition and report where the ignitions landed."""
    cfg = _start(config)
    report = verify_events(cfg)

    path = model_directory(cfg) / EVENTS_FILENAME
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    logger.info(
        "%d ignition(s): median within-day percentile %.3f, %.1f%% at or above the 90th",
        overall["events"],
        overall["median_percentile"],
        100 * overall["share_at_or_above_90"],
    )
    logger.info(
        "Cell effect carries %.1f%% of the log10(p) variance",
        100 * float(report["cell_effect_variance_share"]),
    )
    logger.info("Wrote %s", path)


@app.command("serve")
def serve_command(
    config: ConfigOption = None,
    host: Annotated[str | None, typer.Option("--host", help="Interface to bind.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Port to bind.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on a source change.")] = False,
) -> None:
    """Run the web application."""
    import uvicorn

    from tfire.api import create_app

    cfg = _start(config)
    uvicorn.run(
        "tfire.api:create_app" if reload else create_app(cfg),
        factory=reload,
        host=host or cfg.app.host,
        port=port or cfg.app.port,
        reload=reload,
        workers=1,
    )


@app.command("admin-password")
def admin_password_command() -> None:
    """Hash an admin password and print the environment line to store it in."""
    from tfire.api.auth import PASSWORD_ENV, hash_password

    secret = typer.prompt("Admin password", hide_input=True, confirmation_prompt=True)
    typer.echo(f"\n{PASSWORD_ENV}={hash_password(secret)}\n")
    typer.echo("Put that line in .env, which is git-ignored. The password itself is not stored.")


@app.command("package-runtime")
def package_runtime_command(
    config: ConfigOption = None,
    out: Annotated[
        Path, typer.Option("--out", help="Directory to assemble the runtime subset in.")
    ] = Path("dist/runtime"),
    force: ForceOption = False,
) -> None:
    """Assemble the data a serving container needs, and nothing the build needs."""
    cfg = _start(config)
    package_runtime(cfg, cfg.path(out), force=force)


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
