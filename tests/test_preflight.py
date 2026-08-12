"""Pre-flight wiring tests.

The probes themselves need credentials and network, so they are never executed
here, only the pure logic around them (bbox ordering, result handling, dispatch).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tfire.config import BBoxWGS84, Config
from tfire.preflight import CHECKS, AccessCheck, check_access


def test_bbox_bounds_use_the_earth_engine_ordering() -> None:
    """Earth Engine wants xmin/ymin/xmax/ymax; feeding it N/W/S/E is the bug to catch."""
    bbox = BBoxWGS84(north=46.6, west=10.4, south=45.6, east=12.0)
    assert bbox.as_bounds() == [10.4, 45.6, 12.0, 46.6]


def test_configured_bbox_covers_trentino(config: Config) -> None:
    bbox = config.sources.bbox_wgs84
    # Trento is at roughly 46.07 N, 11.12 E.
    assert bbox.south < 46.07 < bbox.north
    assert bbox.west < 11.12 < bbox.east


@pytest.mark.parametrize(
    "kwargs",
    [
        {"north": 45.0, "west": 10.4, "south": 46.6, "east": 12.0},  # inverted lat
        {"north": 46.6, "west": 12.0, "south": 45.6, "east": 10.4},  # inverted lon
    ],
)
def test_malformed_bbox_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        BBoxWGS84(**kwargs)


def test_every_source_is_registered() -> None:
    assert set(CHECKS) == {"era5", "gee", "landsat", "osm", "worldpop"}


def test_failed_check_is_reported_not_raised(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing probe must return a result so every source still gets tried."""

    def boom(_: Config) -> AccessCheck:
        return AccessCheck("ERA5", ok=False, detail="RuntimeError: no credentials")

    monkeypatch.setitem(CHECKS, "era5", boom)
    (result,) = check_access(config, ["era5"])

    assert not result.ok
    assert "no credentials" in result.detail
