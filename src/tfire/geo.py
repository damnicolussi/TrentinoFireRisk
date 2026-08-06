"""Geospatial helpers shared by the vector-reading stages."""

from __future__ import annotations

import logging
from pathlib import Path

from pyproj import CRS

logger = logging.getLogger(__name__)

# how sure pyproj has to be before it hands back an EPSG code
_EPSG_CONFIDENCE = 70


def verify_crs(actual: CRS | None, expected_crs: str, source: Path) -> None:
    """Raise unless the layer is already in the working CRS. Nothing here reprojects.

    The PAT `.prj` files are ESRI WKT with no authority code, so a string comparison
    against `"EPSG:25832"` fails on data that is in fact EPSG:25832. The code is
    recovered instead, with a pyproj equality test as the fallback.
    """
    if actual is None:
        raise ValueError(f"{source} has no CRS; expected {expected_crs}")

    expected = CRS.from_user_input(expected_crs)
    actual_epsg = actual.to_epsg(min_confidence=_EPSG_CONFIDENCE)
    if actual_epsg != expected.to_epsg() and actual != expected:
        raise ValueError(
            f"CRS mismatch in {source}: expected {expected_crs}, "
            f"got {actual.name!r} (EPSG:{actual_epsg}). Refusing to reproject."
        )
    logger.info("CRS verified as %s (%s) for %s", expected_crs, actual.name, source.name)
