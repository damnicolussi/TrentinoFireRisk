"""Crop the Europe-wide CORINE GeoTIFFs to the Trentino study area.

Reprojects each edition from EPSG:3035 into the working CRS and writes the
result to `data/interim/corine/`, leaving `data/raw/` untouched.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from tfire.config import Config, load_config, setup_logging

logger = logging.getLogger("crop_corine")


def load_cutline(config: Config) -> gpd.GeoSeries:
    """Boundary dissolved to one geometry and buffered by the configured margin."""
    boundary_path = config.path(config.paths.pat_boundary)
    if not boundary_path.is_file():
        raise FileNotFoundError(f"PAT boundary not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError(f"{boundary_path} has no CRS")

    working = boundary.to_crs(config.crs)
    dissolved = working.geometry.union_all()
    buffered = dissolved.buffer(config.corine.margin_m)

    logger.info(
        "Cutline from %s: %d feature(s), buffered by %.0f m",
        boundary_path.name,
        len(boundary),
        config.corine.margin_m,
    )
    return gpd.GeoSeries([buffered], crs=config.crs)


def target_grid(
    cutline: gpd.GeoSeries, source: rasterio.DatasetReader, resolution_m: int
) -> tuple[rasterio.Affine, int, int]:
    """Output transform and shape in the working CRS, snapped to the 500 m grid."""
    left, bottom, right, top = cutline.total_bounds

    # snap outward so the origin lands on a multiple of the grid resolution
    left = np.floor(left / resolution_m) * resolution_m
    bottom = np.floor(bottom / resolution_m) * resolution_m
    right = np.ceil(right / resolution_m) * resolution_m
    top = np.ceil(top / resolution_m) * resolution_m

    pixel = source.res[0]
    width = int(round((right - left) / pixel))
    height = int(round((top - bottom) / pixel))
    return rasterio.Affine(pixel, 0.0, left, 0.0, -pixel, top), width, height


def read_source_window(
    source: rasterio.DatasetReader, cutline: gpd.GeoSeries
) -> tuple[np.ndarray, rasterio.Affine]:
    """Read only the part of the European raster that covers the cutline."""
    in_source_crs = cutline.to_crs(source.crs)
    left, bottom, right, top = in_source_crs.total_bounds

    # one pixel of slack so the warp has neighbours at the edge
    pad = source.res[0]
    window = from_bounds(
        left - pad, bottom - pad, right + pad, top + pad, transform=source.transform
    )
    window = window.round_offsets().round_lengths()
    window = window.intersection(rasterio.windows.Window(0, 0, source.width, source.height))

    data = source.read(1, window=window)
    return data, source.window_transform(window)


@dataclass(frozen=True)
class CropResult:
    year: int
    source_mb: float
    out_mb: float
    height: int
    width: int
    codes: list[int]
    coverage: float


def crop_edition(
    year: int, source_path: Path, out_path: Path, cutline: gpd.GeoSeries, config: Config
) -> CropResult:
    with rasterio.open(source_path) as source:
        nodata = source.nodata
        if nodata is None:
            raise ValueError(f"{source_path} declares no NODATA value; refusing to guess one")

        window_data, window_transform = read_source_window(source, cutline)
        transform, width, height = target_grid(cutline, source, config.resolution_m)

        destination = np.full((height, width), nodata, dtype=source.dtypes[0])
        reproject(
            source=window_data,
            destination=destination,
            src_transform=window_transform,
            src_crs=source.crs,
            src_nodata=nodata,
            dst_transform=transform,
            dst_crs=config.crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )

        profile = source.profile.copy()

    outside = geometry_mask(
        cutline.geometry, out_shape=(height, width), transform=transform, invert=False
    )
    destination[outside] = nodata

    profile.update(
        driver="GTiff",
        crs=config.crs,
        transform=transform,
        width=width,
        height=height,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as destination_file:
        destination_file.write(destination, 1)

    inside = destination[destination != nodata]
    return CropResult(
        year=year,
        source_mb=source_path.stat().st_size / 1e6,
        out_mb=out_path.stat().st_size / 1e6,
        height=height,
        width=width,
        codes=sorted(int(c) for c in np.unique(inside)),
        coverage=float(inside.size) / destination.size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Recrop editions already present.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)

    cutline = load_cutline(config)
    raw_root = config.path(config.paths.corine_raw)
    out_root = config.path(config.paths.corine_out)

    results: list[CropResult] = []
    missing: list[int] = []

    for year, relative in sorted(config.corine.editions.items()):
        source_path = raw_root / relative
        out_path = out_root / f"clc_{year}.tif"

        if not source_path.is_file():
            logger.error("CLC%d: source missing at %s", year, source_path)
            missing.append(year)
            continue

        if out_path.exists() and not args.force:
            logger.info("CLC%d: already cropped, skipping (use --force)", year)
            continue

        logger.info("CLC%d: cropping %s", year, source_path.name)
        results.append(crop_edition(year, source_path, out_path, cutline, config))

    for row in results:
        logger.info(
            "CLC%d: %.1f MB -> %.2f MB, %dx%d, %.1f%% inside cutline, %d classes %s",
            row.year,
            row.source_mb,
            row.out_mb,
            row.height,
            row.width,
            100 * row.coverage,
            len(row.codes),
            row.codes,
        )

    if len(results) > 1:
        shared = set.intersection(*(set(r.codes) for r in results))
        for row in results:
            odd = sorted(set(row.codes) - shared)
            if odd:
                logger.warning("CLC%d has classes no other edition has: %s", row.year, odd)

    if missing:
        logger.error("Missing source rasters for: %s", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
