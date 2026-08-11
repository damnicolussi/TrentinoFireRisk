"""Feature extraction, one module per category."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from tfire.config import Config
from tfire.features.fwi import extract_fwi
from tfire.features.geography import extract_geography
from tfire.features.landcover import extract_landcover
from tfire.features.meteo import extract_meteo
from tfire.features.topography import extract_topography
from tfire.features.vegetation import extract_vegetation

# insertion order is run order: `fwi` reads the table `meteo` writes
EXTRACTORS: dict[str, Callable[[Config, bool], pd.DataFrame]] = {
    "topography": extract_topography,
    "geography": extract_geography,
    "landcover": extract_landcover,
    "vegetation": extract_vegetation,
    "meteo": extract_meteo,
    "fwi": extract_fwi,
}
