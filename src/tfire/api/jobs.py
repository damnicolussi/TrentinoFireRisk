"""Long pipeline commands, run as subprocesses so a retrain cannot sit inside the API process."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tfire.config import Config

logger = logging.getLogger(__name__)

# every action is a sequence of existing CLI verbs; the API adds no pipeline behavior of its own
ACTIONS: Final[dict[str, list[list[str]]]] = {
    "predict": [["predict"]],
    "warm": [["predict"]],
    "refresh-vegetation": [["predict", "--refresh-vegetation"]],
    "refresh-era5": [
        ["fetch-era5"],
        ["extract-features", "--category", "meteo", "--category", "fwi", "--force"],
    ],
    "rebuild-dataset": [["build-dataset", "--force"]],
    "retrain": [["train", "--force"], ["evaluate", "--force", "--sensitivity", "all"]],
}


@dataclass
class Job:
    id: str
    action: str
    commands: list[list[str]]
    log: Path
    started: str
    finished: str | None = None
    returncode: int | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.returncode is None

    def describe(self, tail_lines: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "command": " && ".join(" ".join(argv) for argv in self.commands),
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "running": self.running,
            "log_tail": tail(self.log, tail_lines),
        }


def tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=lines)]


class JobRunner:
    """One slot. A second request while something runs is refused rather than queued."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Job | None:
        with self._lock:
            return self._free()

    def _free(self) -> Job | None:
        """The occupant of the slot, if any. Call with the lock held."""
        if self._current is not None and not self._current.running:
            self._current = None
        return self._current

    def recent(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, action: str, arguments: list[str] | None = None) -> Job:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}, have {sorted(ACTIONS)}")

        steps = ACTIONS[action]
        if arguments and len(steps) > 1:
            raise ValueError(f"{action} runs {len(steps)} steps and takes no arguments")
        commands = [
            [sys.executable, "-m", "tfire.cli", *step, *(arguments or [])] for step in steps
        ]

        job_id = uuid.uuid4().hex[:12]
        directory = self._config.path(self._config.paths.jobs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id,
            action=action,
            commands=commands,
            log=directory / f"{job_id}.log",
            started=datetime.now(UTC).isoformat(timespec="seconds"),
        )

        # claimed under the same lock hold that checks it: the endpoints are sync, so FastAPI runs
        # them in a threadpool and two concurrent posts would otherwise both pass the check
        with self._lock:
            occupant = self._free()
            if occupant is not None:
                raise RuntimeError(f"{occupant.action} is still running")
            self._jobs[job_id] = job
            self._current = job

        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        logger.info("Job %s started: %s", job_id, job.describe(0)["command"])
        return job

    def _run(self, job: Job) -> None:
        """Run the steps in order, stopping at the first failure."""
        code = 0
        with job.log.open("wb") as handle:
            for argv in job.commands:
                job.process = subprocess.Popen(  # noqa: S603
                    argv, cwd=self._config.project_root, stdout=handle, stderr=subprocess.STDOUT
                )
                code = job.process.wait()
                if code:
                    break

        job.returncode = code
        job.finished = datetime.now(UTC).isoformat(timespec="seconds")
        logger.info("Job %s finished with %d", job.id, code)
