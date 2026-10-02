"""Single fires after the record, each against the map of its own day and of the days around it."""

from __future__ import annotations

import datetime as dt
import logging
from datetime import date, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tfire.config import Config
from tfire.features.human import calendar_features
from tfire.grid import GridSpec, load_grid
from tfire.models.danger import DangerClasses, load_danger_classes
from tfire.models.events import history_baseline

if TYPE_CHECKING:
    from tfire.inference import GridScorer

logger = logging.getLogger(__name__)

CASES_REPORT: Final = "cases_2026.md"
CASES_TABLE: Final = "cases_2026.parquet"

Cause = Literal["lightning", "human", "unknown"]
CoordSource = Literal["report", "map", "cell", "municipality", "cadastre"]

WINDOW_DAYS: Final = 15

_LIGHTNING_CODE: Final = 10

_DRIVERS: Final = 3
_DAYS_PER_YEAR: Final = 365.25


class Case(BaseModel):
    """One row of the fire file."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^\S+$")
    date: dt.date
    time: dt.time | None = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    uncertainty_m: float = Field(ge=0)
    coord_source: CoordSource
    place: str = Field(min_length=1)
    municipality: str | None = None
    area_ha: float | None = Field(default=None, ge=0)
    cause: Cause
    source: str | None = None
    notes: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("date", mode="before")
    @classmethod
    def _day_first(cls, value: object) -> object:
        """Italian sources write 31/07/2026; day first, never month first."""
        if isinstance(value, str) and "/" in value:
            return dt.datetime.strptime(value.strip(), "%d/%m/%Y").date()
        return value

    @field_validator("area_ha", "lat", "lon", "uncertainty_m", mode="before")
    @classmethod
    def _decimal_comma(cls, value: object) -> object:
        if isinstance(value, str) and "," in value and "." not in value:
            return value.replace(",", ".")
        return value


def read_cases(path: Path) -> list[Case]:
    """Parse and validate the fire file, naming every bad row rather than the first."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    cases, errors = [], []
    for line, row in enumerate(frame.to_dict("records"), start=2):
        try:
            cases.append(Case.model_validate(row))
        except ValidationError as error:
            fields = ", ".join(str(item["loc"][0]) for item in error.errors() if item["loc"])
            errors.append(f"line {line}: {fields or error}")

    if errors:
        raise ValueError(f"{path} has {len(errors)} bad row(s): " + "; ".join(errors))
    duplicated = sorted({case.id for case in cases if [c.id for c in cases].count(case.id) > 1})
    if duplicated:
        raise ValueError(f"{path} repeats id(s) {duplicated}")
    return cases


def cadastre_cases(config: Config) -> list[Case]:
    """Cadastre fires dated after the modeling record, the ones no fit has seen."""
    from pyproj import Transformer

    fires = pd.read_parquet(config.path(config.paths.fires_out))
    later = fires[fires["ignition_date"].dt.year > config.date_range.end.year]
    to_wgs84 = Transformer.from_crs(config.crs, "EPSG:4326", always_xy=True)

    cases = []
    for row in later.itertuples(index=False):
        lon, lat = to_wgs84.transform(row.x, row.y)
        hour = None if pd.isna(row.start_hour) else int(row.start_hour)
        cases.append(
            Case(
                id=f"cadastre-{row.fire_id}",
                date=row.ignition_date.date(),
                time=None if hour is None else time(hour, 0),
                lat=lat,
                lon=lon,
                uncertainty_m=0.0,
                coord_source="cadastre",
                place=str(row.loc) if not pd.isna(row.loc) else f"fire {row.fire_id}",
                area_ha=None if pd.isna(row.area_ha) else float(row.area_ha),
                cause="lightning" if row.cause == _LIGHTNING_CODE else "unknown",
            )
        )
    return cases


def locate(spec: GridSpec, crs: str, lat: float, lon: float) -> tuple[float, float, int | None]:
    """Grid coordinates of a WGS84 point, and its cell."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return float(x), float(y), spec.point_to_cell(x, y)


def cells_within(
    spec: GridSpec, active: npt.NDArray[np.int64], x: float, y: float, radius_m: float
) -> npt.NDArray[np.int64]:
    """Active cells centered within `radius_m` of the point, its own cell always among them."""
    centers_x, centers_y = spec.cell_center(active)
    near = active[np.hypot(centers_x - x, centers_y - y) <= radius_m]
    own = spec.point_to_cell(x, y)
    if own is not None and own in set(active.tolist()) and own not in set(near.tolist()):
        near = np.append(near, own)
    return np.sort(near)


def _provider(scorer: GridScorer, day: date) -> str:
    if scorer.plan.meteo_from_cache:
        return "cached"
    for piece in scorer.plan.segments:
        if piece.start <= day <= piece.end:
            return str(piece.provider)
    return "unknown"


def _drivers(scorer: GridScorer, frame: pd.DataFrame, position: int) -> list[str]:
    """The largest SHAP contributions on one cell, summed back onto registry names."""
    import shap

    from tfire.models.explain import registry_name
    from tfire.models.trentino import align_columns, design_matrix

    features, _, _ = design_matrix(frame.iloc[[position]].assign(is_fire=False), scorer.registry)
    row = align_columns(features, scorer.columns)
    values = np.asarray(shap.TreeExplainer(scorer.estimator).shap_values(row))[0]

    known = {spec.name for spec in scorer.registry.features}
    summed = pd.Series(values, index=[registry_name(name, known) for name in row.columns])
    summed = summed.groupby(level=0).sum()
    top = summed.reindex(summed.abs().sort_values(ascending=False).index[:_DRIVERS])
    return [f"{name} {value:+.2f}" for name, value in top.items()]


def score_case(
    config: Config,
    case: Case,
    spec: GridSpec,
    active: npt.NDArray[np.int64],
    baseline: pd.Series,
    classes: DangerClasses,
    today: date,
) -> dict[str, Any]:
    """Every measure for one fire, off the shipped model's maps of its surrounding days."""
    from tfire.inference import GridScorer

    x, y, cell = locate(spec, config.crs, case.lat, case.lon)
    if cell is None or cell not in set(active.tolist()):
        raise ValueError(f"{case.id}: {case.lat}, {case.lon} is not on an active cell")
    near = cells_within(spec, active, x, y, case.uncertainty_m)

    days = [
        case.date + timedelta(days=offset)
        for offset in range(-WINDOW_DAYS, WINDOW_DAYS + 1)
        if case.date + timedelta(days=offset) <= today
    ]
    scorer = GridScorer(config, days, today)

    own_scores: dict[date, float] = {}
    measures: dict[str, Any] = {}
    for day in days:
        frame, raw, probability, rank = scorer.day(day)
        position = int(np.flatnonzero(frame["cell_id"].to_numpy() == cell)[0])
        own_scores[day] = float(raw[position])
        if day != case.date:
            continue

        inside = np.isin(frame["cell_id"].to_numpy(), near)
        measures = {
            "within_day_percentile": float(rank[position]),
            "within_day_percentile_max": float(rank[inside].max()),
            "record_percentile": classes.raw_record_percentile(float(raw[position])),
            "danger_class": classes.class_keys[int(classes.classify(probability[position]))],
            "probability": float(probability[position]),
            "return_period_years": 1.0 / (float(probability[position]) * _DAYS_PER_YEAR),
            "drivers": _drivers(scorer, frame, position),
            "provider": _provider(scorer, day),
        }

    series = pd.Series(own_scores)
    return {
        "id": case.id,
        "date": case.date,
        "place": case.place,
        "municipality": case.municipality,
        "cause": case.cause,
        "coord_source": case.coord_source,
        "uncertainty_m": case.uncertainty_m,
        "area_ha": case.area_ha,
        "cell_id": int(cell),
        "cells_in_radius": len(near),
        "night_time": _near_midnight(config, case.time),
        **measures,
        "temporal_percentile": float(series.rank(pct=True)[case.date]),
        "window_days": len(days),
        "forecast_days": sum(_provider(scorer, day) == "forecast" for day in days),
        "baseline_percentile": float(baseline[cell]),
        "baseline_percentile_max": float(baseline.reindex(near).max()),
    }


def _near_midnight(config: Config, moment: time | None) -> bool:
    if moment is None:
        return False
    return (
        moment.hour >= config.fires.near_midnight_start_hour
        or moment.hour < config.fires.near_midnight_end_hour
    )


def verify_cases(
    config: Config, path: Path | None, today: date | None = None
) -> tuple[pd.DataFrame, Path]:
    """Score every fire in the file plus the cadastre's post-record fires, and write the report."""
    stamp = today or date.today()
    cases = cadastre_cases(config) + (read_cases(path) if path else [])
    if not cases:
        raise ValueError("No fire to verify: the file is empty and the cadastre ends in the record")

    spec, grid = load_grid(config)
    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy("int64")
    baseline = history_baseline(config, config.date_range.end.year)
    classes = load_danger_classes(config)

    from tfire.inference import FeatureUnavailableError
    from tfire.sources.forecast import ForecastError

    rows, skipped = [], {}
    for case in sorted(cases, key=lambda item: item.date):
        logger.info("Scoring %s (%s, %s)", case.id, case.date, case.place)
        try:
            rows.append(score_case(config, case, spec, active, baseline, classes, stamp))
        except (FeatureUnavailableError, ForecastError) as error:
            # a weather provider refusing one window should not cost every other fire
            logger.error("Skipping %s: %s", case.id, error)
            skipped[case.id] = str(error)
    if not rows:
        raise ValueError(f"No fire could be scored: {skipped}")

    table = pd.DataFrame(rows)
    calendar = calendar_features(pd.DatetimeIndex(pd.to_datetime(table["date"])))
    table["season"] = calendar["season"].astype(str).to_numpy()

    directory = config.path(config.paths.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table.to_parquet(directory / CASES_TABLE, index=False)
    report = directory / CASES_REPORT
    report.write_text(render_cases(table, config, stamp, skipped), encoding="utf-8")
    logger.info("Wrote %s and %s", report, directory / CASES_TABLE)
    return table, report


def _summary(table: pd.DataFrame, by: str) -> list[list[Any]]:
    rows = []
    for key, part in [("all", table), *table.groupby(by, observed=True)]:
        rows.append(
            [
                key,
                len(part),
                f"{part['within_day_percentile'].median():.3f}",
                f"{(part['within_day_percentile'] >= 0.9).mean():.0%}",
                f"{part['temporal_percentile'].median():.3f}",
                f"{part['baseline_percentile'].median():.3f}",
                f"{(part['baseline_percentile'] >= 0.9).mean():.0%}",
            ]
        )
    return rows


def _weather(provider: str, forecast_days: int) -> str:
    return f"{provider}, {forecast_days} forecast day(s)" if forecast_days else provider


def render_cases(
    table: pd.DataFrame, config: Config, today: date, skipped: dict[str, str] | None = None
) -> str:
    from tfire.report import table as markdown

    events = [
        [
            row["id"],
            f"{row['date']:%Y-%m-%d}",
            row["place"],
            row["cause"],
            f"{row['within_day_percentile']:.3f}",
            f"{row['within_day_percentile_max']:.3f}",
            row["record_percentile"],
            row["danger_class"],
            f"{row['temporal_percentile']:.2f}",
            f"{row['baseline_percentile']:.3f}",
            f"{row['baseline_percentile_max']:.3f}",
            f"{row['probability']:.2e}",
            f"{row['return_period_years']:,.0f}",
            ", ".join(row["drivers"]),
            _weather(row["provider"], int(row["forecast_days"])),
        ]
        for row in table.to_dict("records")
    ]
    summary_headers = (
        "group",
        "n",
        "median day percentile",
        "at or above 90th",
        "median temporal percentile",
        "baseline median",
        "baseline at or above 90th",
    )
    night = table.loc[table["night_time"], "id"].tolist()
    lines = [
        "# Fires after the record",
        "",
        f"Scored on {today:%Y-%m-%d} by the shipped {config.trentino.version} model, which was "
        f"fitted on {config.date_range.start.year}-{config.date_range.end.year}, so none of "
        "these fires was seen in fitting. Percentiles are within the day, on the estimator's "
        "score, at the fire's cell and at the highest cell within its location uncertainty. "
        f"The temporal percentile ranks the fire's day among the {WINDOW_DAYS} days either "
        "side of it at the same cell (days after today are left out).",
        "",
        "The baseline is the fire-history density of "
        f"{config.date_range.start.year}-{config.date_range.end.year} "
        f"({config.history.bandwidth_m / 1000:g} km kernel). It is the same map every day, so "
        "only the within-day measures apply to it.",
        "",
        *markdown(
            (
                "id",
                "date",
                "place",
                "cause",
                "day pct",
                "day pct max in radius",
                "record pct",
                "class",
                "temporal pct",
                "baseline",
                "baseline max in radius",
                "probability",
                "return period (years)",
                "top SHAP",
                "weather",
            ),
            events,
        ),
        "",
        "## By cause",
        "",
        *markdown(summary_headers, _summary(table, "cause")),
        "",
        "## By season",
        "",
        *markdown(summary_headers, _summary(table, "season")),
        "",
    ]
    if night:
        lines += [
            f"Start time between {config.fires.near_midnight_start_hour}:00 and "
            f"{config.fires.near_midnight_end_hour}:00, where the cadastre often carries "
            f"placeholder times: {', '.join(night)}.",
            "",
        ]
    if skipped:
        lines += [
            f"Not scored, {len(skipped)} fire(s), because no weather could be fetched for their "
            "window; rerun once the provider answers:",
            "",
            *[f"- {name}: {reason}" for name, reason in skipped.items()],
            "",
        ]
    stitched = table.loc[table["forecast_days"] > 0, "id"].tolist()
    if stitched:
        lines += [
            "Scored partly on forecast rather than archive weather, to be rerun once the "
            f"archive covers those days: {', '.join(stitched)}.",
            "",
        ]
    return "\n".join(lines)
