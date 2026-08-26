"""The operator surface: sign in, read freshness, switch model version, run a pipeline command."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from tfire.api import service
from tfire.api.auth import SESSION_COOKIE, Sessions, configured_hash, verify_password
from tfire.api.cache import RateLimiter, client_key, warm_window
from tfire.api.jobs import ACTIONS, JobRunner
from tfire.api.state import active_version, available_versions, select_version
from tfire.config import Config

logger = logging.getLogger(__name__)


class Credentials(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class VersionChoice(BaseModel):
    version: str = Field(min_length=1, max_length=64)


class JobRequest(BaseModel):
    action: str
    date: str | None = None
    days: int = Field(default=1, ge=1, le=60)


def build_router(
    config: Config, sessions: Sessions, runner: JobRunner, logins: RateLimiter
) -> APIRouter:
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    def require_session(tfire_session: str | None = Cookie(default=None)) -> str:
        if not sessions.valid(tfire_session):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in first")
        return tfire_session or ""

    guard = Depends(require_session)

    @router.get("/enabled")
    def enabled() -> dict[str, bool]:
        return {"enabled": configured_hash() is not None}

    @router.post("/login")
    def login(credentials: Credentials, request: Request, response: Response) -> dict[str, bool]:
        encoded = configured_hash()
        if encoded is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No admin password is set")

        # per client, not one global counter: a shared counter lets anyone lock the operator out
        # for an hour with ten wrong guesses
        wait = logins.take(client_key(request))
        if wait:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, f"Too many attempts, retry in {wait:.0f} s"
            )

        if not verify_password(credentials.password, encoded):
            logger.warning("Rejected an admin sign-in attempt")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong password")

        # marking the cookie Secure over plain HTTP would make the browser drop it, so it
        # follows the scheme the request actually arrived on, proxy header included
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        response.set_cookie(
            SESSION_COOKIE,
            sessions.open(),
            httponly=True,
            samesite="lax",
            secure=scheme == "https",
            max_age=int(sessions.ttl_seconds),
        )
        return {"ok": True}

    @router.post("/logout")
    def logout(
        response: Response, tfire_session: str | None = Cookie(default=None)
    ) -> dict[str, bool]:
        sessions.close(tfire_session)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @router.get("/status", dependencies=[guard])
    def admin_status() -> dict[str, Any]:
        return {
            **service.status(config),
            "warm_window": [day.isoformat() for day in warm_window(config)],
        }

    @router.get("/versions", dependencies=[guard])
    def versions() -> dict[str, Any]:
        return {"active": active_version(config), "available": available_versions(config)}

    @router.post("/versions", dependencies=[guard])
    def choose_version(choice: VersionChoice) -> dict[str, Any]:
        try:
            select_version(config, choice.version)
        except ValueError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
        return {"active": active_version(config)}

    @router.get("/jobs", dependencies=[guard])
    def jobs() -> dict[str, Any]:
        return {
            "actions": sorted(ACTIONS),
            "jobs": [job.describe(config.app.job_log_tail_lines) for job in runner.recent()],
        }

    @router.post("/jobs", dependencies=[guard])
    def start_job(request: JobRequest) -> dict[str, Any]:
        arguments = []
        if request.action == "warm":
            window = warm_window(config)
            arguments = ["--date", window[0].isoformat(), "--days", str(len(window))]
        elif request.date:
            arguments = ["--date", request.date, "--days", str(request.days)]

        try:
            job = runner.start(request.action, arguments)
        except ValueError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        return job.describe(config.app.job_log_tail_lines)

    @router.get("/jobs/{job_id}", dependencies=[guard])
    def job_detail(job_id: str) -> dict[str, Any]:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job {job_id}")
        return job.describe(config.app.job_log_tail_lines)

    return router
