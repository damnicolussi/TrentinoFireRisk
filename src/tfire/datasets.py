"""Join the labeled samples against every feature block into the model-ready training table."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfire.config import Config
from tfire.features.human import calendar_features
from tfire.features.landcover import nearest_edition
from tfire.features.registry import Registry, load_registry, validate_frame
from tfire.features.vegetation import preceding_composite
from tfire.models.mesogeos import MODEL_FILENAME, PROB_COLUMN, predict_prob

logger = logging.getLogger(__name__)

DECADE_EDGES = (1984, 1990, 2000, 2010, 2020)


def _read(config: Config, relative: Path) -> pd.DataFrame:
    return pd.read_parquet(config.path(relative))


def _join(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Left join that refuses to change the row count.

    A many-to-many key silently duplicates samples and every downstream count stays plausible,
    so the arity is asserted rather than trusted.
    """
    rows = len(left)
    merged = left.merge(right, on=on, how="left", validate="m:1")
    if len(merged) != rows:
        raise ValueError(f"Join on {on} changed the row count: {rows} -> {len(merged)}")
    return merged


def nearest_backbone(weights: pd.DataFrame) -> pd.DataFrame:
    """The single ERA5-Land cell each grid cell leans on most, ties to the lowest `era5_id`."""
    ordered = weights.sort_values(["cell_id", "weight", "era5_id"], ascending=[True, False, True])
    return ordered.drop_duplicates("cell_id")[["cell_id", "era5_id"]].reset_index(drop=True)


def interpolate_meteo(
    samples: pd.DataFrame, weights: pd.DataFrame, meteo: pd.DataFrame
) -> pd.DataFrame:
    """Bilinear interpolation of the daily aggregates from the backbone onto the sample cells.

    Only the linear aggregates go through this. The FWI sub-indices are accumulators carrying
    months of drying history, so a spatial average of them is not the index of any real point.
    """
    columns = [name for name in meteo.columns if name not in ("era5_id", "date")]

    pairs = samples[["sample_id", "cell_id", "date"]].merge(weights, on="cell_id", how="left")
    pairs = pairs.merge(meteo, on=["era5_id", "date"], how="left", validate="m:1")

    mass = pairs.groupby("sample_id")["weight"].sum()
    drift = float((mass - 1.0).abs().max())
    if drift > 1e-6:
        raise ValueError(f"Interpolation weights do not sum to 1 per sample, off by {drift:.3g}")

    values = pairs[columns].to_numpy("float64")
    weighted = pd.DataFrame(values * pairs["weight"].to_numpy()[:, None], columns=columns)
    weighted["sample_id"] = pairs["sample_id"].to_numpy()
    totals = weighted.groupby("sample_id", sort=False).sum()

    # a null in any of the four neighbors makes the weighted sum a partial one, which pandas
    # would hand back as a plausible number
    incomplete = pairs.loc[np.isnan(values).any(axis=1), "sample_id"].unique()
    if incomplete.size:
        logger.error("%d sample(s) have an incomplete backbone neighborhood", incomplete.size)
        totals.loc[totals.index.isin(incomplete), :] = np.nan

    return totals.astype("float32").reset_index()


def attach_landcover(samples: pd.DataFrame, config: Config) -> pd.DataFrame:
    landcover = _read(config, config.paths.landcover_out)
    samples["clc_edition"] = nearest_edition(samples["date"], config.corine.editions)
    return _join(samples, landcover, ["cell_id", "clc_edition"])


def attach_population(samples: pd.DataFrame, config: Config) -> pd.DataFrame:
    population = _read(config, config.paths.human_population_out)
    samples["worldpop_year"] = nearest_edition(samples["date"], config.human.worldpop_years)
    return _join(samples, population, ["cell_id", "worldpop_year"])


def attach_vegetation(samples: pd.DataFrame, config: Config) -> pd.DataFrame:
    vegetation = _read(config, config.paths.vegetation_out)
    months = sorted(vegetation["date"].dt.date.unique())
    samples["veg_composite_date"] = preceding_composite(samples["date"], months)
    vegetation = vegetation.rename(columns={"date": "veg_composite_date"})
    return _join(samples, vegetation, ["cell_id", "veg_composite_date"])


def attach_mesogeos_prob(samples: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Score the assembled rows with the stacking base model, if one has been trained."""
    model_path = config.path(config.paths.mesogeos_model_dir) / MODEL_FILENAME
    if not model_path.is_file():
        logger.warning(
            "No base model at %s, building without %s. Run `tfire train-mesogeos` first.",
            model_path,
            PROB_COLUMN,
        )
        return samples

    samples[PROB_COLUMN] = predict_prob(samples, config)
    logger.info(
        "%s: mean %.4f on the negatives, %.4f on the positives",
        PROB_COLUMN,
        float(samples.loc[~samples["is_fire"], PROB_COLUMN].mean()),
        float(samples.loc[samples["is_fire"], PROB_COLUMN].mean()),
    )
    return samples


def assemble(config: Config) -> pd.DataFrame:
    samples = _read(config, config.paths.samples_out)
    logger.info("Spine: %d samples, %d positive", len(samples), int(samples["is_fire"].sum()))

    for relative in (
        config.paths.topography_out,
        config.paths.geography_out,
        config.paths.human_out,
    ):
        samples = _join(samples, _read(config, relative), ["cell_id"])

    samples = attach_landcover(samples, config)
    samples = attach_population(samples, config)
    samples = attach_vegetation(samples, config)

    weights = _read(config, config.paths.era5_weights_out)
    meteo = _read(config, config.paths.meteo_out)
    samples = _join(samples, interpolate_meteo(samples, weights, meteo), ["sample_id"])

    samples = _join(samples, nearest_backbone(weights), ["cell_id"])
    samples = _join(samples, _read(config, config.paths.fwi_out), ["era5_id", "date"])
    samples = samples.drop(columns="era5_id")

    calendar = calendar_features(pd.DatetimeIndex(samples["date"].unique()))
    samples = _join(samples, calendar, ["date"])

    return attach_mesogeos_prob(samples, config)


def decade_labels(years: pd.Series) -> pd.Series:
    """Bin years into the decades of the record, each labeled by the years it actually holds."""
    edges = np.asarray(DECADE_EDGES)
    start = edges[np.searchsorted(edges, years.to_numpy(), side="right") - 1]
    frame = pd.DataFrame({"year": years.to_numpy(), "start": start})
    spans = frame.groupby("start")["year"].agg(["min", "max"])
    names = {row.Index: f"{row.min}-{row.max}" for row in spans.itertuples()}
    return pd.Series([names[value] for value in start], index=years.index, dtype="string")


def quality_report(frame: pd.DataFrame, registry: Registry, config: Config) -> dict[str, Any]:
    """Missingness per feature per decade, distributions, and the class balance."""
    features = registry.present(frame)
    decade = decade_labels(frame["date"].dt.year)

    balance = []
    missingness: dict[str, dict[str, float]] = {}
    for label, block in frame.groupby(decade, observed=True):
        positives = int(block["is_fire"].sum())
        balance.append(
            {
                "decade": label,
                "rows": len(block),
                "positives": positives,
                "negatives": len(block) - positives,
                "positive_share": positives / len(block),
            }
        )
        missingness[str(label)] = {
            spec.name: float(block[spec.name].isna().mean()) for spec in features
        }

    distributions = []
    for spec in features:
        values = frame[spec.name]
        numeric = pd.to_numeric(values, errors="coerce")
        distributions.append(
            {
                "name": spec.name,
                "category": spec.category,
                "temporal": spec.temporal,
                "missing": float(values.isna().mean()),
                "min": None if numeric.isna().all() else float(numeric.min()),
                "max": None if numeric.isna().all() else float(numeric.max()),
                "mean": None if numeric.isna().all() else float(numeric.mean()),
                "std": None if numeric.isna().all() else float(numeric.std()),
            }
        )

    positives = int(frame["is_fire"].sum())
    return {
        "rows": len(frame),
        "features": len(features),
        "positives": positives,
        "negatives": len(frame) - positives,
        "duplicate_cell_dates": int(frame.duplicated(["cell_id", "date"]).sum()),
        "balance_by_decade": balance,
        "missingness_by_decade": missingness,
        "distributions": distributions,
        "provenance": {
            "samples_per_clc_edition": frame["clc_edition"].value_counts().sort_index().to_dict(),
            "samples_per_worldpop_year": frame["worldpop_year"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "samples_without_composite": int(frame["veg_composite_date"].isna().sum()),
            "usable_ndvi_share": float(frame["ndvi"].notna().mean()),
            "mean_composite_age_days": float(frame["age_days"].mean()),
        },
    }


def validate_dataset(frame: pd.DataFrame, registry: Registry, config: Config) -> None:
    validate_frame(frame, registry, config)

    if len(frame) != config.dataset.expected_rows:
        logger.error("%d row(s), expected %d", len(frame), config.dataset.expected_rows)

    present = registry.present(frame)
    optional = sum(spec.optional for spec in present)
    expected = config.dataset.expected_features + optional
    if len(present) != expected:
        logger.error("%d feature(s), expected %d", len(present), expected)

    duplicates = int(frame.duplicated(["cell_id", "date"]).sum())
    if duplicates:
        logger.error("%d duplicated (cell_id, date) pair(s)", duplicates)

    composite = frame["veg_composite_date"].dt.to_period("M")
    leaked = int((composite >= frame["date"].dt.to_period("M")).sum())
    if leaked:
        logger.error("%d sample(s) drew a composite from their own month or later", leaked)

    positives = int(frame["is_fire"].sum())
    if positives != config.sampling.expected_positives:
        logger.error("%d positive(s), expected %d", positives, config.sampling.expected_positives)


def build_dataset(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `trentino_dataset.parquet` and the data quality report beside it."""
    out = config.path(config.paths.dataset_out)
    report_out = config.path(config.paths.quality_report_out)
    if out.exists() and report_out.exists() and not force:
        logger.info("Outputs already exist, skipping (use --force to rebuild): %s", out)
        return pd.read_parquet(out)

    registry = load_registry()
    frame = assemble(config)
    frame["clc_edition"] = frame["clc_edition"].astype("int16")
    frame["worldpop_year"] = frame["worldpop_year"].astype("int16")

    validate_dataset(frame, registry, config)
    frame = frame[[spec.name for spec in registry.specs if spec.name in frame.columns]]

    report = quality_report(frame, registry, config)
    for row in report["balance_by_decade"]:
        logger.info(
            "%s | %6d rows | %4d positives (%.2f%%)",
            row["decade"],
            row["rows"],
            row["positives"],
            100 * row["positive_share"],
        )
    logger.info(
        "Vegetation: %.1f%% of samples carry an ndvi, mean composite age %.1f days",
        100 * report["provenance"]["usable_ndvi_share"],
        report["provenance"]["mean_composite_age_days"],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    report_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %d rows x %d columns to %s", len(frame), frame.shape[1], out)
    logger.info("Wrote the data quality report to %s", report_out)
    return frame
