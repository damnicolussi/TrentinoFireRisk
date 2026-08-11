"""Earth Engine ERA5-Land retrieval: band order, window tiling, lattice cropping."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from tfire.config import Config
from tfire.sources.era5land import (
    MAX_GEE_BANDS,
    Lattice,
    bbox_lattice,
    crop_indices,
    half_hours,
    read_window,
    windows,
)


def tweak(config: Config, **meteo: object) -> Config:
    return config.model_copy(update={"meteo": config.meteo.model_copy(update=meteo)})


def geotiff(values: np.ndarray, transform: Affine) -> bytes:
    count, height, width = values.shape
    with rasterio.MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as destination:
            destination.write(values.astype("float32"))
        payload: bytes = memory.read()
    return payload


def test_bbox_lattice_matches_the_cds_axes(config: Config) -> None:
    """Both ends of the box sit on grid lines, where float division lands just under."""
    lattice = bbox_lattice(config)
    assert lattice.latitudes.tolist() == pytest.approx([46.6 - 0.1 * step for step in range(11)])
    assert lattice.longitudes.tolist() == pytest.approx([10.4 + 0.1 * step for step in range(17)])
    assert (lattice.n_rows, lattice.n_cols, lattice.n_cells) == (11, 17, 187)


@pytest.mark.parametrize(
    ("year", "half"),
    [(2000, 1), (2000, 2), (2001, 1), (2001, 2), (1984, 1)],
)
def test_windows_tile_a_half_year_exactly_once(config: Config, year: int, half: int) -> None:
    spans = windows(config, year, half)
    hours = sum((end - start).total_seconds() / 3600 for start, end in spans)
    assert hours == half_hours(year, half)

    for (_, end), (following, _) in zip(spans[:-1], spans[1:], strict=True):
        assert end == following

    span = config.meteo.gee_window_hours
    assert all((end - start).total_seconds() / 3600 <= span for start, end in spans)
    assert span * len(config.meteo.variables) <= MAX_GEE_BANDS


def test_windows_reject_a_span_over_the_band_cap(config: Config) -> None:
    oversized = MAX_GEE_BANDS // len(config.meteo.variables) + 1
    with pytest.raises(ValueError, match="over Earth Engine's limit"):
        windows(tweak(config, gee_window_hours=oversized), 2000, 1)


def test_read_window_keeps_hours_outermost_and_variables_inner() -> None:
    """`toBands` is image-major, so a transposed reshape would swap variables between hours."""
    lattice = Lattice(np.array([46.6, 46.5]), np.array([10.4, 10.5, 10.6]))
    n_hours, n_variables = 3, 2
    bands = np.arange(n_hours * n_variables, dtype="float32")
    values = np.broadcast_to(bands[:, None, None], (bands.size, 2, 3))
    payload = geotiff(values, Affine(0.1, 0, 10.35, 0, -0.1, 46.65))

    window = read_window(payload, lattice, n_hours, n_variables)
    assert window.shape == (n_hours, n_variables, 2, 3)
    for hour in range(n_hours):
        for variable in range(n_variables):
            assert np.all(window[hour, variable] == hour * n_variables + variable)


def test_read_window_rejects_a_band_count_that_is_not_hours_by_variables() -> None:
    lattice = Lattice(np.array([46.6]), np.array([10.4]))
    payload = geotiff(np.zeros((5, 1, 1)), Affine(0.1, 0, 10.35, 0, -0.1, 46.65))
    with pytest.raises(ValueError, match="expected 3 hours x 2 variables"):
        read_window(payload, lattice, 3, 2)


def test_crop_indices_finds_the_lattice_in_a_raster_grown_outward(config: Config) -> None:
    """Earth Engine snaps the region to whole tiles, so the origin comes back moved."""
    lattice = bbox_lattice(config)
    rows, columns = crop_indices(Affine(0.1, 0, 10.35, 0, -0.1, 46.75), 18, 12, lattice)
    assert rows.tolist() == list(range(1, 12))
    assert columns.tolist() == list(range(17))


def test_crop_indices_rejects_a_raster_missing_a_lattice_point(config: Config) -> None:
    lattice = bbox_lattice(config)
    with pytest.raises(ValueError, match="latitudes"):
        crop_indices(Affine(0.1, 0, 10.35, 0, -0.1, 46.75), 18, 6, lattice)
