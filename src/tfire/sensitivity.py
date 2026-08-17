"""Rebuild the training table under a disturbed assumption and re-score the model on it."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from tfire.config import Config
from tfire.datasets import assemble
from tfire.evaluation import scores
from tfire.features.registry import load_registry
from tfire.features.vegetation import extract_vegetation
from tfire.models.trentino import SPECS, design_matrix, fit_and_predict, training_mask
from tfire.sampling import build_samples

logger = logging.getLogger(__name__)

Variant = Callable[[Config, Path], Config]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    question: str
    prepare: Variant


def _redirect(config: Config, directory: Path, **paths: str) -> Config:
    """A copy of the config writing the named artifacts under `directory` instead."""
    directory.mkdir(parents=True, exist_ok=True)
    updated = config.paths.model_copy(
        update={key: directory / value for key, value in paths.items()}
    )
    return config.model_copy(update={"paths": updated})


def shift_near_midnight(config: Config, directory: Path) -> Config:
    """Move every positive logged either side of midnight onto the adjacent day.

    Which day a fire logged next to midnight belongs to is the one thing the cadastre cannot
    settle: 22:00 and later become the next day, before 02:00 the previous one.
    """
    variant = _redirect(config, directory, samples_out="samples.parquet")
    out = config.path(variant.paths.samples_out)
    if not out.exists():
        samples = pd.read_parquet(config.path(config.paths.samples_out))
        fires = pd.read_parquet(config.path(config.paths.fires_out))
        shifted_samples(samples, fires, config).to_parquet(out, index=False)
    return variant


def shifted_samples(samples: pd.DataFrame, fires: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The sample table with every near-midnight positive moved across the day boundary.

    A shifted row can land on a `(cell_id, date)` another row already holds. The positive wins
    that collision, since dropping it instead would quietly weaken the very label under test.
    """
    hour = samples["fire_id"].map(fires.set_index("fire_id")["start_hour"])
    late = samples["is_fire"] & (hour >= config.fires.near_midnight_start_hour)
    early = samples["is_fire"] & (hour < config.fires.near_midnight_end_hour)

    shift = pd.to_timedelta(late.astype("int8") - early.astype("int8"), unit="D")
    moved = samples.assign(date=samples["date"] + shift)

    inside = (moved["date"] >= pd.Timestamp(config.date_range.start)) & (
        moved["date"] <= pd.Timestamp(config.date_range.end)
    )
    kept = (
        moved.loc[inside]
        .sort_values("is_fire", ascending=False, kind="stable")
        .drop_duplicates(["cell_id", "date"])
        .sort_values(["date", "cell_id"], kind="stable", ignore_index=True)
    )
    logger.info(
        "near_midnight: %d positive(s) shifted, %d dropped outside the record, %d as collisions",
        int(late.sum() + early.sum()),
        int((~inside).sum()),
        int(inside.sum()) - len(kept),
    )
    return kept


def negative_ratio(ratio: int) -> Variant:
    """Redraw the negatives at `ratio` per positive, same seed, same exclusion set."""

    def prepare(config: Config, directory: Path) -> Config:
        variant = _redirect(
            config,
            directory,
            samples_out="samples.parquet",
            exclusions_out="exclusions.parquet",
        )
        variant = variant.model_copy(
            update={"sampling": variant.sampling.model_copy(update={"negative_ratio": ratio})}
        )
        build_samples(variant)
        return variant

    return prepare


def vegetation_climatology(config: Config, directory: Path) -> Config:
    """Fill a cell-month with no usable observation from its own monthly normal, not with NaN."""
    variant = _redirect(
        config,
        directory,
        vegetation_out="vegetation.parquet",
        vegetation_climatology_out="vegetation_climatology.parquet",
        vegetation_missingness_out="vegetation_missingness.parquet",
    )
    variant = variant.model_copy(
        update={
            "vegetation": variant.vegetation.model_copy(
                update={"no_observation_strategy": "climatology"}
            )
        }
    )

    # the extractor rewrites the climatology it also reads, so the variant gets its own copy of
    # the canonical one rather than a normal computed from already-filled values
    source = config.path(config.paths.vegetation_climatology_out)
    target = config.path(variant.paths.vegetation_climatology_out)
    if not target.exists():
        shutil.copyfile(source, target)

    extract_vegetation(variant)
    return variant


VARIANTS: Final[dict[str, VariantSpec]] = {
    spec.name: spec
    for spec in (
        VariantSpec(
            "near_midnight_shift",
            "does the label survive moving the fires either side of midnight to the other day?",
            shift_near_midnight,
        ),
        VariantSpec(
            "negative_ratio_5",
            "is 1:10 doing the work, or would 1:5 give the same ranking?",
            negative_ratio(5),
        ),
        VariantSpec(
            "negative_ratio_20",
            "does a harder imbalance change anything beyond the base rate?",
            negative_ratio(20),
        ),
        VariantSpec(
            "vegetation_climatology",
            "do the cells with no Landsat observation matter enough to fill them?",
            vegetation_climatology,
        ),
    )
}


def run_variant(spec: VariantSpec, config: Config, tuning: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the table this variant disturbs and score the tuned model on it."""
    directory = config.path(config.paths.sensitivity_dir) / spec.name
    variant = spec.prepare(config, directory)

    frame = assemble(variant)
    registry = load_registry()
    frame = frame[[item.name for item in registry.specs if item.name in frame.columns]]

    features, labels, years = design_matrix(frame, registry)
    train = training_mask(years, variant)
    fitted = fit_and_predict(SPECS["xgboost"], features, labels, years, train, variant, tuning)

    pooled = scores(labels[train], fitted.out_of_fold)
    held = scores(labels[~train], fitted.holdout)
    logger.info(
        "%-24s | pooled AUPRC %.4f | holdout AUPRC %.4f",
        spec.name,
        pooled["auprc"],
        held["auprc"],
    )
    return {
        "question": spec.question,
        "rows": len(frame),
        "positives": int(labels.sum()),
        "features": len(fitted.columns),
        "pooled_out_of_fold": pooled,
        "holdout": held,
    }


def deltas(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, float]:
    """Variant minus canonical, on the metrics the report compares."""
    return {
        f"{split}_{metric}": variant[split][metric] - baseline[split][metric]
        for split in ("pooled_out_of_fold", "holdout")
        for metric in ("auprc", "auroc", "lift")
    }


def run(
    config: Config, names: list[str], tuning: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Every requested variant, each with its metrics and its distance from the canonical run."""
    results: dict[str, Any] = {}
    for name in names:
        spec = VARIANTS[name]
        logger.info("Sensitivity: %s (%s)", spec.name, spec.question)
        measured = run_variant(spec, config, tuning)
        measured["delta"] = deltas(baseline, measured)
        results[name] = measured

    if results:
        worst = max(results.values(), key=lambda item: abs(float(item["delta"]["holdout_auprc"])))
        logger.info(
            "Largest holdout AUPRC move across %d variant(s): %+.4f",
            len(results),
            worst["delta"]["holdout_auprc"],
        )
    return results


def resolve(requested: list[str] | None) -> list[str]:
    """Variant names to run, in declaration order. `all` or nothing selects every one."""
    if not requested or "all" in requested:
        return list(VARIANTS)
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}. Choose from {list(VARIANTS)}")
    return [name for name in VARIANTS if name in set(requested)]
