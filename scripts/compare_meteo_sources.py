"""Compare the served weather against the cached ERA5-Land backbone over the same days.

Writes `reports/meteo_sources.md`. Both columns come out of the same daily aggregator, so a
difference is the provider and the resolution, not the arithmetic. Precipitation and wind are
the fields Open-Meteo serves from ERA5 at 0.25 degrees rather than ERA5-Land at 0.1, which is
the skew this measures.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config, load_config, setup_logging
from tfire.features.meteo import add_lag_features, aggregate_daily, to_frame
from tfire.sources import forecast
from tfire.sources.era5land import read_lattice

logger = logging.getLogger("compare_meteo_sources")

REPORT_FILENAME = "meteo_sources.md"

# a month either side of the fire season, so the comparison is not one weather regime
DEFAULT_SPANS: list[tuple[date, date]] = [
    (date(2024, 6, 1), date(2024, 6, 30)),
    (date(2024, 11, 1), date(2024, 11, 30)),
]

# the served fields, in the order the report lists them. The first three are ERA5-Land on both
# sides; the rest come off the coarser grid remotely.
COLUMNS = (
    "temp_mean",
    "temp_max",
    "rh_mean",
    "pres_mean",
    "precip_sum",
    "wind_speed_mean",
    "wind_speed_max",
)

SAME_RESOLUTION = frozenset({"temp_mean", "temp_max", "rh_mean"})

# relative bias past which a field is called out rather than tabulated and left alone
BIAS_THRESHOLD = 0.05


@dataclass(frozen=True)
class Comparison:
    span: str
    column: str
    days: int
    cached_mean: float
    served_mean: float
    bias: float
    mae: float
    correlation: float

    @property
    def relative_bias(self) -> float:
        return self.bias / abs(self.cached_mean) if self.cached_mean else float("nan")


def served_daily(config: Config, start: date, end: date) -> pd.DataFrame:
    """The same span through the serving path, with the lag columns its features need.

    The window opens early so `precip_cum30` and the moving averages have their history; only
    the requested days come back.
    """
    lattice = read_lattice(config)
    lead = timedelta(days=config.meteo.longest_window_days)
    fields, segments = forecast.fetch_span(config, lattice, start - lead, end)
    logger.info(
        "Served %s to %s through %s",
        start,
        end,
        " then ".join(f"{piece.provider}" for piece in segments),
    )

    dates, columns = aggregate_daily(fields, config)
    add_lag_features(columns, config)
    frame = to_frame(dates, columns, lattice)
    span: pd.DataFrame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return span


def cached_daily(config: Config, start: date, end: date) -> pd.DataFrame:
    frame = pd.read_parquet(config.path(config.paths.meteo_out))
    span: pd.DataFrame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return span


def _correlation(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compare(config: Config, start: date, end: date) -> list[Comparison]:
    cached = cached_daily(config, start, end)
    served = served_daily(config, start, end)

    keys = ["era5_id", "date"]
    joined = cached.merge(served, on=keys, how="inner", suffixes=("_cached", "_served"))
    if joined.empty:
        raise ValueError(f"No overlapping cell-day between the two sources over {start} to {end}")

    span = f"{start} to {end}"
    rows = []
    for column in COLUMNS:
        left = joined[f"{column}_cached"].to_numpy(dtype="float64")
        right = joined[f"{column}_served"].to_numpy(dtype="float64")
        rows.append(
            Comparison(
                span=span,
                column=column,
                days=int(joined["date"].nunique()),
                cached_mean=float(left.mean()),
                served_mean=float(right.mean()),
                bias=float((right - left).mean()),
                mae=float(np.abs(right - left).mean()),
                correlation=_correlation(left, right),
            )
        )
    logger.info("%s: %d cell-day(s) over %d day(s)", span, len(joined), rows[0].days)
    return rows


def render(config: Config, results: list[Comparison]) -> Path:
    root = config.path(config.paths.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = root / REPORT_FILENAME

    header = "| span | field | days | cached mean | served mean | bias | rel. bias | MAE | r |"
    lines = [
        "# Served weather against the cached backbone",
        "",
        "The cached column is ERA5-Land at 0.1 degrees out of Earth Engine, the served column "
        "is the same span fetched through the operational path and put through the same daily "
        "aggregator. Open-Meteo's `era5_seamless` serves temperature and dewpoint from "
        "ERA5-Land and everything else from ERA5 at 0.25 degrees, so the last four rows of "
        "each block are the ones that can move.",
        "",
        header,
        "|" + "---|" * 9,
    ]
    for row in results:
        mark = "" if row.column in SAME_RESOLUTION else " *"
        lines.append(
            f"| {row.span} | `{row.column}`{mark} | {row.days} | {row.cached_mean:.3f} | "
            f"{row.served_mean:.3f} | {row.bias:+.3f} | {row.relative_bias:+.1%} | "
            f"{row.mae:.3f} | {row.correlation:.3f} |"
        )

    lines += ["", "\\* served from ERA5 at 0.25 degrees rather than ERA5-Land at 0.1.", ""]
    lines += _findings(results)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _findings(results: list[Comparison]) -> list[str]:
    skewed = sorted(
        {row.column for row in results if row.column not in SAME_RESOLUTION}
        & {row.column for row in results if abs(row.relative_bias) > BIAS_THRESHOLD}
    )
    agreed = sorted(
        {row.column for row in results if row.column not in SAME_RESOLUTION} - set(skewed)
    )

    lines = ["## What this says", ""]
    if skewed:
        worst = {
            column: max((row.relative_bias for row in results if row.column == column), key=abs)
            for column in skewed
        }
        lines += [
            "Served past the backbone, "
            + ", ".join(f"`{column}` runs {worst[column]:+.0%}" for column in skewed)
            + " against the record the model was trained on. A feature that arrives "
            "systematically higher than its training distribution shifts every prediction "
            "that depends on it, which is what the bias correction is for.",
            "",
        ]
    if agreed:
        lines += [
            "The coarser grid costs nothing measurable on "
            + ", ".join(f"`{column}`" for column in agreed)
            + ", so no correction is warranted there.",
            "",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--span",
        action="append",
        metavar="START:END",
        help="ISO date pair to compare; repeatable. Defaults to June and November 2024.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)

    spans = DEFAULT_SPANS
    if args.span:
        spans = [
            (date.fromisoformat(piece.split(":")[0]), date.fromisoformat(piece.split(":")[1]))
            for piece in args.span
        ]

    results: list[Comparison] = []
    for start, end in spans:
        results.extend(compare(config, start, end))

    out = render(config, results)
    logger.info("Wrote %s", out)

    for row in results:
        if row.column not in SAME_RESOLUTION and abs(row.relative_bias) > BIAS_THRESHOLD:
            logger.warning(
                "%s %s: served mean is %+.1f%% against the backbone",
                row.span,
                row.column,
                100 * row.relative_bias,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
