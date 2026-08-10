"""Feature extraction, one module per category."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from tfire.config import Config
from tfire.features.geography import extract_geography
from tfire.features.landcover import extract_landcover
from tfire.features.topography import extract_topography

EXTRACTORS: dict[str, Callable[[Config, bool], pd.DataFrame]] = {
    "geography": extract_geography,
    "landcover": extract_landcover,
    "topography": extract_topography,
}
