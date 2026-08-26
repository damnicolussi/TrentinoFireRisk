"""The read path of the web application, the staleness guard, and the runtime manifest."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tfire.api import create_app, service
from tfire.api.auth import PASSWORD_ENV, hash_password
from tfire.api.state import select_version
from tfire.config import Config
from tfire.inference import (
    DRIVER_COLUMNS,
    PROBABILITY_BAND,
    RANK_BAND,
    active_cells,
    model_fingerprint,
)
from tfire.models.danger import CLASS_KEYS, DangerClasses, reference_days
from tfire.packaging import runtime_paths
from tfire.sources.era5land import bbox_lattice, read_lattice

from .conftest import requires_built, requires_model

DAY = date(2019, 8, 12)
PASSWORD = "test-password"


def rooted(config: Config, tmp_path: Path, **paths: Path) -> Config:
    """The real config with some output directories redirected under `tmp_path`."""
    moved = {name: tmp_path / relative for name, relative in paths.items()}
    return config.model_copy(update={"paths": config.paths.model_copy(update=moved)})


def write_map(config: Config, day: date, cell_ids: list[int], probability: list[float]) -> None:
    """A risk map on disk the way `write_day` leaves one, without scoring anything."""
    paths = service.outputs(config, day)
    paths["table"].parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "cell_id": np.asarray(cell_ids, dtype="int32"),
            PROBABILITY_BAND: np.asarray(probability, dtype="float32"),
            RANK_BAND: np.linspace(0.0, 100.0, len(cell_ids), dtype="float32"),
            **{name: np.zeros(len(cell_ids), dtype="float32") for name in DRIVER_COLUMNS},
        }
    )
    frame.to_parquet(paths["table"], index=False)
    paths["raster"].touch()
    paths["meta"].write_text(
        json.dumps(
            {
                "date": day.isoformat(),
                "model_version": config.trentino.version,
                "model_fingerprint": model_fingerprint(config),
                "cells": len(active_cells(config)),
                "sources": ["cached"],
            }
        ),
        encoding="utf-8",
    )


def sidecar(config: Config, day: date) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(
        service.outputs(config, day)["meta"].read_text(encoding="utf-8")
    )
    return payload


def rewrite_sidecar(config: Config, day: date, **changes: object) -> None:
    payload = {**sidecar(config, day), **changes}
    service.outputs(config, day)["meta"].write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def served(config: Config, tmp_path: Path) -> Config:
    """A config whose risk maps and overlays live in a temporary directory."""
    return rooted(config, tmp_path, risk_dir=Path("risk"), overlay_dir=Path("risk/overlays"))


def test_a_clicked_point_resolves_to_the_cell_it_sits_in(served: Config) -> None:
    """The coordinates a summary publishes have to come back as the cells they describe.

    Longitude and latitude are both plain floats, so an inverted pair reaches `point_to_cell`
    without complaint and answers about somewhere else.
    """
    requires_built(served, served.paths.grid_out)

    grid = pd.read_parquet(served.path(served.paths.grid_out))
    cell_ids = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()[::2000].tolist()
    write_map(served, DAY, cell_ids, np.linspace(1e-6, 1e-3, len(cell_ids)).tolist())

    for entry in service.day_summary(served, DAY)["top_cells"]:
        found = service.query_cell(served, DAY, entry["lon"], entry["lat"])
        assert found is not None
        assert found["cell_id"] == entry["cell_id"]
        assert found["probability"] == pytest.approx(entry["probability"])
        assert service.query_cell(served, DAY, entry["lat"], entry["lon"]) is None


def test_the_overlay_sits_exactly_on_the_bounds_it_is_published_with(config: Config) -> None:
    """The transform and the published corners have to describe the same rectangle.

    Leaflet stretches the PNG over whatever `bounds` say, so a transform covering a different
    extent slides the whole province a few hundred metres and still looks like a map.
    """
    from rasterio.warp import transform_bounds

    requires_built(config, config.paths.grid_out, config.paths.grid_spec_out)

    spec = service.grid_spec(config)
    geometry = service.overlay_geometry(config)

    expected = transform_bounds(spec.crs, service.WGS84, *spec.bounds)
    assert geometry.bounds_wgs84 == pytest.approx(expected)

    # Leaflet takes the corners latitude first, and a flipped pair still draws a rectangle
    south, west, north, east = (value for pair in geometry.leaflet_bounds for value in pair)
    assert (west, south, east, north) == pytest.approx(expected)
    assert south < north and west < east

    mercator = transform_bounds(spec.crs, service.WEB_MERCATOR, *spec.bounds)
    top_left = geometry.transform * (0, 0)
    bottom_right = geometry.transform * (geometry.width, geometry.height)
    assert top_left == pytest.approx((mercator[0], mercator[3]))
    assert bottom_right == pytest.approx((mercator[2], mercator[1]))

    assert max(geometry.width, geometry.height) <= config.app.overlay_max_px
    assert (geometry.width, geometry.height) == (spec.n_cols * 8, spec.n_rows * 8)


@pytest.mark.parametrize("offset", [-1e-12, 0.0, 1e-12])
def test_a_probability_at_a_break_lands_in_the_class_the_break_opens(offset: float) -> None:
    classes = DangerClasses(
        breaks=[0.1, 0.2, 0.3, 0.4],
        percentiles=[90.0, 99.0, 99.9, 99.99],
        class_keys=list(CLASS_KEYS),
        reference="test",
        reference_years=[2015, 2024],
        rows=1,
        model_version="v1",
        config_sha256="0" * 64,
    )

    assigned = classes.classify([0.05, 0.1, 0.15, 0.2, 0.35, 0.4, 0.9, np.nan])
    assert list(assigned) == [0, 1, 1, 2, 3, 4, 4, -1]
    # a break belongs to the class above it, and the ordering never inverts
    assert classes.classify([break_value + offset for break_value in classes.breaks]).tolist() == (
        [1, 2, 3, 4] if offset >= 0 else [0, 1, 2, 3]
    )


def test_the_record_percentile_reads_off_the_rung_below_the_probability() -> None:
    """A panel that says 99.9 where the truth is 99.8 is plausible, quotable and wrong."""
    # rung i is the (i / 10)th percentile, so this ladder puts value v at percentile v / 10
    ladder = [float(rung) for rung in range(1001)]
    classes = DangerClasses(
        breaks=[100.0, 500.0, 900.0, 990.0],
        percentiles=[90.0, 99.0, 99.9, 99.99],
        class_keys=list(CLASS_KEYS),
        reference="test",
        reference_years=[2015, 2024],
        rows=len(ladder),
        model_version="v1",
        config_sha256="0" * 64,
        quantiles=ladder,
    )

    # exactly on a rung, between two rungs, and past either end
    assert classes.record_percentile(0.0) == 0.0
    assert classes.record_percentile(500.0) == 50.0
    assert classes.record_percentile(500.5) == 50.0
    assert classes.record_percentile(999.0) == 99.9
    assert classes.record_percentile(1000.0) == 100.0
    assert classes.record_percentile(1e9) == 100.0
    assert classes.record_percentile(-1.0) == 0.0


def test_an_artifact_cut_before_the_quantile_ladder_reports_no_record_percentile() -> None:
    classes = DangerClasses(
        breaks=[0.1, 0.2, 0.3, 0.4],
        percentiles=[90.0, 99.0, 99.9, 99.99],
        class_keys=list(CLASS_KEYS),
        reference="test",
        reference_years=[2015, 2024],
        rows=1,
        model_version="v1",
        config_sha256="0" * 64,
    )
    assert classes.record_percentile(0.35) is None


def test_a_date_no_provider_covers_is_refused_with_its_bounds(served: Config) -> None:
    client = TestClient(create_app(served))
    first, last = service.date_bounds(served)

    for day in (first - timedelta(days=1), last + timedelta(days=1)):
        response = client.get(f"/api/risk/{day.isoformat()}")
        assert response.status_code == 422
        assert response.json()["first_date"] == first.isoformat()
        assert response.json()["last_date"] == last.isoformat()

    assert client.get("/api/risk/not-a-date").status_code == 400


def test_a_map_written_under_different_conditions_is_recomputed(served: Config) -> None:
    """Every one of these leaves a file that is present, readable and wrong."""
    requires_built(served, served.paths.grid_out)

    write_map(served, DAY, [0, 1, 2], [1e-6, 1e-5, 1e-4])
    assert not service.is_stale(served, DAY)

    for change in (
        {"model_version": "v0"},
        {"model_fingerprint": "0" * 16},
        {"cells": 1},
        # a date the backbone now covers, still carrying the 0.25 degree provider it was
        # scored on before the cached tables were extended over it
        {"sources": ["open-meteo-archive:2019-08-12..2019-08-12"]},
    ):
        rewrite_sidecar(served, DAY, **change)
        assert service.is_stale(served, DAY), change
        write_map(served, DAY, [0, 1, 2], [1e-6, 1e-5, 1e-4])

    table = pd.read_parquet(service.outputs(served, DAY)["table"])
    table.drop(columns=[DRIVER_COLUMNS[0]]).to_parquet(
        service.outputs(served, DAY)["table"], index=False
    )
    assert service.is_stale(served, DAY)

    service.outputs(served, DAY)["raster"].unlink()
    assert service.is_stale(served, DAY)


def test_repainting_the_map_moves_the_overlay_address_and_its_token(served: Config) -> None:
    """The PNGs are served immutable, so anything that changes a pixel has to change the URL."""
    requires_built(served, served.paths.grid_out)
    requires_model(served, "danger_classes.json")

    classes = service.danger_classes(served)
    before = (
        service.overlay_path(served, DAY, "classes", classes),
        service.overlay_token(served, classes),
    )

    # a retrain in place: same version, same breaks, different estimator bytes
    calibrator = (
        served.path(served.paths.trentino_model_dir) / served.trentino.version / "calibrator.json"
    )
    original = calibrator.read_bytes()
    try:
        calibrator.write_bytes(original + b"\n")
        after = (
            service.overlay_path(served, DAY, "classes", classes),
            service.overlay_token(served, classes),
        )
    finally:
        calibrator.write_bytes(original)

    assert after[0] != before[0]
    assert after[1] != before[1]

    # the two palettes never share a file either, whatever the breaks say
    assert service.overlay_path(served, DAY, "rank", classes) != before[0]


def test_a_danger_artifact_cut_for_another_model_or_grid_is_refused(
    config: Config, tmp_path: Path
) -> None:
    """Wrong breaks recolor the whole province and nothing about the map looks broken."""
    requires_model(config)

    directory = tmp_path / "trentino" / config.trentino.version
    directory.mkdir(parents=True)
    for name in ("model.json", "metrics.json", "calibrator.json"):
        source = config.path(config.paths.trentino_model_dir) / config.trentino.version / name
        (directory / name).write_bytes(source.read_bytes())

    moved = config.model_copy(
        update={
            "paths": config.paths.model_copy(update={"trentino_model_dir": tmp_path / "trentino"})
        }
    )
    rows = moved.grid.expected_active_cells * len(reference_days(moved))

    def write(**changes: object) -> None:
        fields: dict[str, Any] = {
            "breaks": [1e-5, 1e-4, 1e-3, 1e-2],
            "percentiles": [90.0, 99.0, 99.9, 99.99],
            "class_keys": list(CLASS_KEYS),
            "reference": "test",
            "reference_years": [2015, 2024],
            "rows": rows,
            "model_version": moved.trentino.version,
            "config_sha256": moved.digest(),
        }
        DangerClasses(**{**fields, **changes}).write(directory / "danger_classes.json")

    write()
    assert service.danger_classes(moved).rows == rows

    for change in ({"model_version": "v0"}, {"rows": rows - 1}):
        write(**change)
        with pytest.raises(service.StaleArtifactError):
            service.danger_classes(moved)


def test_the_admin_surface_needs_a_session(served: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PASSWORD_ENV, hash_password(PASSWORD))
    client = TestClient(create_app(served))

    assert client.get("/api/admin/versions").status_code == 401

    assert client.post("/api/admin/login", json={"password": "wrong"}).status_code == 401
    assert client.get("/api/admin/versions").status_code == 401

    assert client.post("/api/admin/login", json={"password": PASSWORD}).status_code == 200
    assert client.get("/api/admin/versions").status_code == 200

    client.post("/api/admin/logout")
    assert client.get("/api/admin/versions").status_code == 401


def test_the_operators_version_choice_reaches_the_model_a_request_loads(
    config: Config, tmp_path: Path
) -> None:
    """Reporting the chosen version while loading another one's artifacts is silent and wrong."""
    requires_model(config)

    versions = tmp_path / "trentino"
    served = rooted(
        config,
        tmp_path,
        risk_dir=Path("risk"),
        overlay_dir=Path("risk/overlays"),
        app_state_out=Path("app_state.json"),
        trentino_model_dir=Path("trentino"),
    )

    source = config.path(config.paths.trentino_model_dir) / config.trentino.version
    rows = served.grid.expected_active_cells * len(reference_days(served))

    # named off the config rather than hardcoded: which version ships moves between releases,
    # and the property under test is that the choice reaches the artifacts, not what it is called
    shipped = config.trentino.version
    other = "rollback"
    breaks_of = {shipped: [1e-5, 1e-4, 1e-3, 1e-2], other: [2e-5, 2e-4, 2e-3, 2e-2]}
    for name, breaks in breaks_of.items():
        directory = versions / name
        directory.mkdir(parents=True)
        for artifact in ("model.json", "metrics.json", "calibrator.json"):
            (directory / artifact).write_bytes((source / artifact).read_bytes())
        DangerClasses(
            breaks=breaks,
            percentiles=[90.0, 99.0, 99.9, 99.99],
            class_keys=list(CLASS_KEYS),
            reference="test",
            reference_years=[2015, 2024],
            rows=rows,
            model_version=name,
            config_sha256="0" * 64,
        ).write(directory / "danger_classes.json")

    client = TestClient(create_app(served))
    assert client.get("/api/meta").json()["danger"]["breaks"] == breaks_of[shipped]

    select_version(served, other)
    meta = client.get("/api/meta").json()
    assert meta["model_version"] == other
    assert meta["danger"]["breaks"] == breaks_of[other]
    assert client.get("/healthz").json()["model_version"] == other


def test_the_runtime_manifest_covers_every_input_the_service_opens(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path resolved at request time but missing from the manifest ships a container that 500s.

    Recorded by field name rather than by value, because that is the level `runtime_paths` is
    written at: it lists `paths.X` entries, and a new read of `paths.Y` has to be declared there.
    """
    from tfire.inference import predict_days

    requires_model(config)
    requires_built(config, *(path for path, required in runtime_paths(config).items() if required))

    served = rooted(
        config,
        tmp_path,
        risk_dir=Path("risk"),
        overlay_dir=Path("risk/overlays"),
        app_state_out=Path("app_state.json"),
    )

    field_of = {Path(value): name for name, value in served.paths.model_dump().items()}
    touched: set[str] = set()
    original = Config.path

    def recording(self: Config, relative: Path) -> Path:
        name = field_of.get(Path(relative))
        if name is not None:
            touched.add(name)
        return original(self, relative)

    monkeypatch.setattr(Config, "path", recording)

    predict_days(served, DAY, days=1, force=True)
    service.day_summary(served, DAY)
    service.render_overlay(served, DAY, "classes")
    service.fires_on(served, DAY)
    service.boundary_geojson(served)
    service.status(served)
    service.about(served)

    monkeypatch.undo()

    shipped = [Path(relative) for relative in runtime_paths(served)]
    declared = {
        name
        for value, name in field_of.items()
        if any(value == relative or value in relative.parents for relative in shipped)
    }
    # written, not read: the service's own output, and the cache a live forecast fills
    outputs = {
        "risk_dir",
        "overlay_dir",
        "jobs_dir",
        "app_state_out",
        "runtime_manifest_out",
        "forecast_cache",
        "operational_dir",
        "frontend_dir",
    }
    # the ERA5 downloads are resolved to look for a cached lattice and are not shipped, so the
    # arithmetic fallback has to agree with them wherever one is present
    assert touched - declared - outputs == {"era5_raw"}
    assert read_lattice(served) == bbox_lattice(served)


def test_the_project_page_counts_the_cadastre_not_the_sampled_positives(
    config: Config, tmp_path: Path
) -> None:
    """The record ends in 2024 while the cadastre file runs past it, and the sampler drops the
    ignitions that land on a cell the model never scores. Either one, reported under the label
    the page carries, is a wrong number that still looks plausible."""
    fires = tmp_path / "fires.parquet"
    pd.DataFrame(
        {
            "fire_id": [1, 2, 3],
            "ignition_date": pd.to_datetime(["1983-12-31", "2000-06-01", "2025-01-02"]),
        }
    ).to_parquet(fires, index=False)

    rerooted = rooted(config, tmp_path, fires_out=Path("fires.parquet"))
    assert service.about(rerooted)["record"]["cadastre_ignitions"] == 1
