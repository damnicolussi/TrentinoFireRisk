"""What keeps a public URL from being a way to fill a disk or pin a CPU."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from fastapi import Request

from tfire.config import Config

logger = logging.getLogger(__name__)

_HOUR_S = 3600.0


def client_key(request: Request) -> str:
    """Who to count a request against. Behind a proxy that is the forwarded address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@dataclass
class RateLimiter:
    """A sliding hourly window per client, counting only the expensive requests."""

    per_hour: int

    def __post_init__(self) -> None:
        self._seen: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def take(self, client: str, now: float | None = None) -> float:
        """Seconds the client has to wait, `0.0` when the request may proceed."""
        if self.per_hour <= 0:
            return 0.0
        stamp = now if now is not None else time.monotonic()
        with self._lock:
            window = self._seen[client]
            while window and stamp - window[0] > _HOUR_S:
                window.popleft()
            if len(window) >= self.per_hour:
                return _HOUR_S - (stamp - window[0])
            window.append(stamp)
            return 0.0

    def forget(self, client: str) -> None:
        with self._lock:
            self._seen.pop(client, None)


def warm_window(config: Config, today: date | None = None) -> list[date]:
    """The dates kept on disk at all times, so the common case never computes."""
    stamp = today or date.today()
    first, last = config.app.warm_window_days
    return [stamp + timedelta(days=offset) for offset in range(first, last + 1)]


def _day_of(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem.split("_", 1)[1][:10])
    except (IndexError, ValueError):
        return None


def evict(config: Config, today: date | None = None) -> list[date]:
    """Drop the least recently read days until the risk cache is back under its cap."""
    directory = config.path(config.paths.risk_dir)
    if not directory.is_dir():
        return []

    protected = set(warm_window(config, today))
    sizes: dict[date, int] = {}
    atimes: dict[date, float] = {}
    files: dict[date, list[Path]] = {}

    for path in [*directory.glob("risk_*.*"), *config.path(config.paths.overlay_dir).glob("*.png")]:
        day = _day_of(path)
        if day is None or not path.is_file():
            continue
        stat = path.stat()
        sizes[day] = sizes.get(day, 0) + stat.st_size
        atimes[day] = max(atimes.get(day, 0.0), stat.st_atime)
        files.setdefault(day, []).append(path)

    budget = int(config.app.cache_max_gb * 1e9)
    total = sum(sizes.values())
    if total <= budget:
        return []

    dropped = []
    for day in sorted(sizes, key=lambda d: atimes[d]):
        if total <= budget:
            break
        if day in protected:
            continue
        for path in files[day]:
            path.unlink(missing_ok=True)
        total -= sizes[day]
        dropped.append(day)

    if dropped:
        logger.info(
            "Evicted %d day(s) to bring the risk cache to %.1f GB", len(dropped), total / 1e9
        )
    return dropped
