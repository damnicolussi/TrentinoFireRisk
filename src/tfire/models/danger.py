"""Absolute danger-class breaks, so a legend means the same thing in December as in August."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from tfire.config import Config

logger = logging.getLogger(__name__)

DANGER_FILENAME: Final = "danger_classes.json"

CLASS_KEYS: Final = ("very_low", "low", "moderate", "high", "very_high")

# rungs of the quantile ladder the cell panel reads a record percentile off, so the finest
# step it can report is a tenth of a percent
QUANTILE_RUNGS: Final = 1001


@dataclass(frozen=True)
class DangerClasses:
    """Breaks on the calibrated probability, plus what they were derived from."""

    breaks: list[float]
    percentiles: list[float]
    class_keys: list[str]
    reference: str
    reference_years: list[int]
    rows: int
    model_version: str
    config_sha256: str
    quantiles: list[float] | None = None
    mean_probability: float | None = None

    def classify(self, probability: npt.ArrayLike) -> npt.NDArray[np.int8]:
        """Class index per value, `-1` where the probability is missing."""
        values = np.asarray(probability, dtype="float64")
        index = np.searchsorted(np.asarray(self.breaks), values, side="right").astype("int8")
        return np.where(np.isnan(values), np.int8(-1), index)

    def record_percentile(self, probability: float) -> float | None:
        """Share of the reference record this probability sits at or above, in percent."""
        if self.quantiles is None:
            return None
        rung = int(np.searchsorted(np.asarray(self.quantiles), probability, side="right")) - 1
        steps = len(self.quantiles) - 1
        return round(100.0 * min(max(rung, 0), steps) / steps, 1)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> DangerClasses:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def reference_days(config: Config) -> list[date]:
    """Every `danger_reference_stride`-th day of the years the shipped estimator never saw."""
    first = date(config.trentino.test_years_start, 1, 1)
    last = config.date_range.end
    if first >= last:
        raise ValueError(f"test_years_start {first.year} leaves no holdout years before {last}")

    span = (last - first).days + 1
    stride = config.trentino.danger_reference_stride
    return [first + timedelta(days=offset) for offset in range(0, span, stride)]


def _reference_scores(config: Config) -> npt.NDArray[np.float64]:
    """Calibrated probabilities over the whole grid, on a sample of holdout days."""
    from tfire.inference import GridScorer

    days = reference_days(config)
    scorer = GridScorer(config, days)
    logger.info("Scoring %d whole day(s) for the danger-class reference", len(days))

    blocks = []
    for index, day in enumerate(days):
        _, probability, _ = scorer.day(day)
        blocks.append(probability.astype("float32"))
        if index and index % 50 == 0:
            logger.info("  %d/%d days", index, len(days))
    return np.concatenate(blocks).astype("float64")


def build_danger_classes(config: Config, force: bool = False) -> DangerClasses:
    from tfire.inference import model_directory

    path = model_directory(config) / DANGER_FILENAME
    if path.is_file() and not force:
        logger.info("%s already exists, use --force to rebuild", path)
        return DangerClasses.read(path)

    scores = _reference_scores(config)

    percentiles = list(config.trentino.danger_percentiles)
    breaks = [float(value) for value in np.percentile(scores, percentiles)]
    if sorted(breaks) != breaks or len(set(breaks)) != len(breaks):
        raise ValueError(f"Percentiles {percentiles} give non-increasing breaks {breaks}")

    years = [config.trentino.test_years_start, config.date_range.end.year]
    classes = DangerClasses(
        breaks=breaks,
        percentiles=percentiles,
        class_keys=list(CLASS_KEYS[: len(breaks) + 1]),
        reference=f"full grid, every day of {years[0]}-{years[1]}"
        if config.trentino.danger_reference_stride == 1
        else f"full grid, every {config.trentino.danger_reference_stride} days "
        f"of {years[0]}-{years[1]}",
        reference_years=years,
        rows=int(scores.size),
        model_version=config.trentino.version,
        config_sha256=config.digest(),
        quantiles=[
            float(value) for value in np.percentile(scores, np.linspace(0, 100, QUANTILE_RUNGS))
        ],
        mean_probability=float(scores.mean()),
    )
    classes.write(path)

    assigned = classes.classify(scores)
    counts = np.bincount(assigned[assigned >= 0], minlength=len(classes.class_keys))
    logger.info(
        "Danger breaks from %d reference row(s): %s | class shares %s | mean %.3e",
        classes.rows,
        ", ".join(f"{value:.3e}" for value in breaks),
        ", ".join(f"{count / scores.size:.1%}" for count in counts),
        classes.mean_probability,
    )
    return classes


def load_danger_classes(config: Config) -> DangerClasses:
    from tfire.inference import model_directory

    path = model_directory(config) / DANGER_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run `tfire danger-classes` first.")
    return DangerClasses.read(path)


def summarize(classes: DangerClasses, probability: npt.NDArray[Any]) -> dict[str, int]:
    """Cells per danger class, for a day's summary panel."""
    assigned = classes.classify(probability)
    counts = np.bincount(assigned[assigned >= 0], minlength=len(classes.class_keys))
    return dict(zip(classes.class_keys, (int(count) for count in counts), strict=True))
