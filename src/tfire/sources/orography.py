"""ERA5-Land's own terrain height on the backbone lattice, from ECMWF's invariant geopotential.

The backbone temperatures belong to this height, not to a DEM averaged to 0.1 degrees.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import requests

from tfire.config import Config

if TYPE_CHECKING:
    from tfire.sources.era5land import Lattice

logger = logging.getLogger(__name__)

STANDARD_GRAVITY: Final = 9.80665

_DOWNLOAD_TIMEOUT_S: Final = 300

# lattice coordinates are decimal tenths; the file stores them as float32
_COORDINATE_TOLERANCE: Final = 1e-3


def fetch_orography(config: Config) -> None:
    """Download the invariant file once. It never changes, so a cached copy is final."""
    path = config.path(config.paths.orography_raw)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading the ERA5-Land invariant geopotential to %s", path)
    response = requests.get(config.meteo.orography_url, timeout=_DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    partial = path.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.rename(path)


def backbone_orography(config: Config, lattice: Lattice) -> npt.NDArray[np.float64]:
    """Terrain height of every backbone cell in meters, in `era5_id` order."""
    import xarray as xr

    fetch_orography(config)
    with xr.open_dataset(config.path(config.paths.orography_raw)) as dataset:
        geopotential = dataset["z"].squeeze(drop=True)
        longitudes = np.mod(lattice.longitudes, 360.0)
        picked = geopotential.sel(
            latitude=lattice.latitudes, longitude=longitudes, method="nearest"
        )
        if not (
            np.allclose(picked["latitude"], lattice.latitudes, atol=_COORDINATE_TOLERANCE)
            and np.allclose(picked["longitude"], longitudes, atol=_COORDINATE_TOLERANCE)
        ):
            raise ValueError("The invariant file is not on the backbone's 0.1 degree lattice")
        heights = picked.to_numpy() / STANDARD_GRAVITY
    return np.asarray(heights, dtype="float64").reshape(-1)
