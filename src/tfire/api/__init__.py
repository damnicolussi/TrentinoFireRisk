"""The FastAPI application: public risk endpoints, the project page's numbers, and the map."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tfire.api import service
from tfire.api.admin import build_router
from tfire.api.auth import Sessions
from tfire.api.cache import RateLimiter, client_key, evict
from tfire.api.jobs import JobRunner
from tfire.api.service import CLASS_COLORS, RANK_COLORS, DateOutOfRangeError
from tfire.api.state import active_config, active_version
from tfire.config import Config, load_config
from tfire.inference import FeatureUnavailableError
from tfire.sources.forecast import ForecastError

logger = logging.getLogger(__name__)

_IMMUTABLE = "public, max-age=604800, immutable"


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{value!r} is not a date in YYYY-MM-DD form"
        ) from error


def create_app(config: Config | None = None) -> FastAPI:
    base = config or load_config()

    def current() -> Config:
        """The config a request works against, with the operator's version choice applied."""
        return active_config(base)

    sessions = Sessions(ttl_seconds=base.app.session_ttl_minutes * 60)
    runner = JobRunner(base)
    predictions = RateLimiter(per_hour=base.app.uncached_requests_per_hour)
    logins = RateLimiter(per_hour=base.app.login_attempts_per_hour)

    app = FastAPI(
        title="Trentino Fire Risk",
        description="Daily wildfire ignition risk for the province of Trento.",
        version=base.trentino.version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.exception_handler(FeatureUnavailableError)
    def unavailable(request: Request, error: FeatureUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "date": error.day.isoformat(),
                "features": error.features,
                "reason": error.reason,
            },
        )

    @app.exception_handler(ForecastError)
    def upstream(request: Request, error: ForecastError) -> JSONResponse:
        # the weather provider refused or is throttling: nothing is wrong with the request, so
        # it is worth retrying rather than a 4xx telling the caller to change something
        logger.warning("Weather provider failed: %s", error)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"reason": str(error), "retry": True},
        )

    @app.exception_handler(DateOutOfRangeError)
    def out_of_range(request: Request, error: DateOutOfRangeError) -> JSONResponse:
        first, last = service.date_bounds(base)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "reason": str(error),
                "first_date": first.isoformat(),
                "last_date": last.isoformat(),
            },
        )

    def serve_day(day: date, request: Request) -> None:
        """Apply the rate limit only when the request would actually have to compute."""
        if not service.is_stale(current(), day):
            return
        if not base.app.on_demand:
            raise FeatureUnavailableError(
                day, ["risk"], "no risk map is stored for this date and on-demand scoring is off."
            )
        wait = predictions.take(client_key(request))
        if wait:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"This date has to be computed and you are over the hourly limit. "
                f"Retry in {wait / 60:.0f} minute(s).",
            )
        service.ensure_day(current(), day)
        evict(base)

    @app.get("/healthz", tags=["service"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model_version": active_version(base)}

    @app.get("/api/meta", tags=["public"])
    def meta() -> dict[str, Any]:
        active = current()
        first, last = service.date_bounds(base)
        classes = service.danger_classes(active)
        geometry = service.overlay_geometry(base)
        return {
            "first_date": first.isoformat(),
            "last_date": last.isoformat(),
            "model_version": active_version(base),
            "default_language": base.app.default_language,
            "overlay": {
                "bounds": geometry.leaflet_bounds,
                "opacity": base.app.overlay_opacity,
                "width": geometry.width,
                "height": geometry.height,
            },
            "map": {
                "min_zoom": base.app.min_zoom,
                "max_zoom": base.app.max_zoom,
                "margin_deg": base.app.bounds_margin_deg,
                "basemaps": [basemap.model_dump() for basemap in base.app.basemaps],
            },
            "danger": {
                "class_keys": classes.class_keys,
                "breaks": classes.breaks,
                "percentiles": classes.percentiles,
                "rank_percentiles": list(service.RANK_PERCENTILES),
                "token": service.overlay_token(active, classes),
                "colors": list(CLASS_COLORS[: len(classes.class_keys)]),
                "rank_colors": list(RANK_COLORS[: len(classes.class_keys)]),
                "reference_years": classes.reference_years,
            },
        }

    @app.get("/api/risk/{day}", tags=["public"])
    def risk(day: str, request: Request) -> dict[str, Any]:
        stamp = _parse_day(day)
        serve_day(stamp, request)
        return service.day_summary(current(), stamp)

    @app.get("/api/risk/{day}/overlay.png", tags=["public"])
    def overlay(day: str, request: Request, palette: str = "classes") -> FileResponse:
        stamp = _parse_day(day)
        if palette not in service.PALETTES:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Unknown palette {palette!r}, have {sorted(service.PALETTES)}",
            )
        serve_day(stamp, request)
        return FileResponse(
            service.render_overlay(current(), stamp, palette),
            media_type="image/png",
            headers={"Cache-Control": _IMMUTABLE},
        )

    @app.get("/api/risk/{day}/cell", tags=["public"])
    def cell(day: str, lon: float, lat: float, request: Request) -> dict[str, Any]:
        stamp = _parse_day(day)
        serve_day(stamp, request)
        found = service.query_cell(current(), stamp, lon, lat)
        if found is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "That point is outside the modeled province"
            )
        return found

    @app.get("/api/risk/{day}/fires", tags=["public"])
    def fires(day: str) -> dict[str, Any]:
        return service.fires_on(base, _parse_day(day))

    @app.get("/api/boundary.geojson", tags=["public"])
    def boundary() -> Response:
        return JSONResponse(service.boundary_geojson(base), headers={"Cache-Control": _IMMUTABLE})

    @app.get("/api/status", tags=["public"])
    def status_endpoint() -> dict[str, Any]:
        return service.status(base)

    @app.get("/api/about", tags=["public"])
    def about() -> dict[str, Any]:
        return service.about(base)

    @app.get("/api/figures/{name}", tags=["public"])
    def figure(name: str) -> FileResponse:
        path = base.path(base.paths.figures_dir) / name
        if "/" in name or ".." in name or path.suffix != ".png" or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No figure {name}")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": _IMMUTABLE})

    app.include_router(build_router(base, sessions, runner, logins))

    frontend = base.path(base.paths.frontend_dir)
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        logger.warning("No frontend at %s, serving the API only", frontend)

    return app
