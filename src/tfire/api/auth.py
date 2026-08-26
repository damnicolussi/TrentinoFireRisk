"""Password hashing and the in-process session table behind the admin surface."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

PASSWORD_ENV: Final = "TFIRE_ADMIN_PASSWORD"
SESSION_COOKIE: Final = "tfire_session"

_SCHEME: Final = "scrypt"
_N: Final = 2**14
_R: Final = 8
_P: Final = 1
_SALT_BYTES: Final = 16

# crypt(3) separates these fields with `$`, but this value travels through an env file that
# docker compose interpolates, where `$16384` reads as a variable and the hash arrives truncated.
# `:` cannot appear in base64.
_SEPARATOR: Final = ":"


def hash_password(password: str) -> str:
    """The string to put in the environment. Salt and parameters travel with the digest."""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P)
    encoded = (base64.b64encode(value).decode("ascii") for value in (salt, derived))
    return _SEPARATOR.join([_SCHEME, str(_N), str(_R), str(_P), *encoded])


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = encoded.split(_SEPARATOR)
        if scheme != _SCHEME:
            raise ValueError(f"unsupported scheme {scheme!r}")
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(digest)),
        )
    except (ValueError, TypeError) as error:
        logger.error("%s is not a valid password hash: %s", PASSWORD_ENV, error)
        return False
    return secrets.compare_digest(derived, base64.b64decode(digest))


def configured_hash() -> str | None:
    """`None` disables the admin surface rather than leaving it unprotected."""
    return os.environ.get(PASSWORD_ENV) or None


@dataclass
class Sessions:
    ttl_seconds: float

    def __post_init__(self) -> None:
        self._issued: dict[str, float] = {}
        self._lock = threading.Lock()

    def open(self, now: float | None = None) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._issued[token] = (now if now is not None else time.monotonic()) + self.ttl_seconds
        return token

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not token:
            return False
        stamp = now if now is not None else time.monotonic()
        with self._lock:
            expiry = self._issued.get(token)
            if expiry is None:
                return False
            if expiry <= stamp:
                del self._issued[token]
                return False
            return True

    def close(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._issued.pop(token, None)
