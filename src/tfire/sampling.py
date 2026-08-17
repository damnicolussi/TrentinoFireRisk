"""Three-level labeling and the negative draw that produce `samples.parquet`."""

from __future__ import annotations

import logging
from typing import Any, Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.grid import GridSpec, load_grid

logger = logging.getLogger(__name__)

LEVEL_POLYGON: Final = 2
LEVEL_BUFFER: Final = 3


def event_dates(fires: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """First and last day of each event, both at midnight."""
    start = fires["ignition_date"]
    end = fires["end_datetime"].dt.normalize().fillna(start)
    return start, end.where(end >= start, start)


def prepare_fires(fires: pd.DataFrame, spec: GridSpec) -> pd.DataFrame:
    """Attach the ignition cell and the event span to every dated fire."""
    dated = fires.loc[fires["ignition_date"].notna()].copy()
    if len(dated) < len(fires):
        logger.warning("%d fire(s) have no ignition date and are ignored", len(fires) - len(dated))

    dated["cell_id"] = spec.points_to_cells(dated["x"], dated["y"])
    dated["start_date"], dated["end_date"] = event_dates(dated)

    span = (dated["end_date"] - dated["start_date"]).dt.days
    logger.info(
        "Event span: median %.0f day(s), mean %.1f, max %d", span.median(), span.mean(), span.max()
    )
    return dated


def build_positive_rows(fires: pd.DataFrame, grid: pd.DataFrame, config: Config) -> pd.DataFrame:
    """One row per cell-day that saw an ignition inside the modeling window.

    Fires whose 500 m cell centers outside the province are dropped, since no feature
    stage covers an inactive cell. Fires sharing a cell-day collapse onto the largest,
    which is what keeps `(cell_id, date)` unique for the training table.
    """
    day = fires["start_date"]
    window = fires.loc[
        day.between(pd.Timestamp(config.date_range.start), pd.Timestamp(config.date_range.end))
    ]
    logger.info(
        "Fires inside %s to %s: %d", config.date_range.start, config.date_range.end, len(window)
    )

    active = grid.loc[grid["is_trentino"], "cell_id"].to_numpy()
    inside = np.isin(window["cell_id"], active)
    if not inside.all():
        logger.warning(
            "Dropped %d positive(s) whose cell center falls outside the boundary: %s",
            int((~inside).sum()),
            window.loc[~inside, "fire_id"].tolist(),
        )
    window = window.loc[inside]

    largest_first = window.sort_values("area_ha", ascending=False, kind="stable")
    positives = largest_first.groupby(["cell_id", "start_date"], as_index=False).agg(
        fire_id=("fire_id", "first"), n_fires=("fire_id", "size")
    )
    collapsed = len(window) - len(positives)
    if collapsed:
        logger.info("Collapsed %d fire(s) onto a cell-day already held by a larger one", collapsed)

    return positives.rename(columns={"start_date": "date"})


def exclusion_pairs(
    fires: pd.DataFrame, polygons: gpd.GeoDataFrame, spec: GridSpec, config: Config
) -> pd.DataFrame:
    """Every `(cell_id, date)` a fire makes unusable as a negative, tagged by level.

    Level 2 is the burned polygon over the event's own days. Level 3 adds a ring of
    `buffer_cells` and `buffer_days` on each side, mapped from the buffered geometry
    rather than from a box around it. Both levels are emitted, so a zero buffer still
    yields level 2, and a cell-day can appear once per fire that reaches it.
    """
    geometry = polygons.set_index("fire_id")["geometry"]
    radius = config.sampling.buffer_cells * config.resolution_m
    slack = pd.Timedelta(days=config.sampling.buffer_days)

    fire_ids: list[npt.NDArray[Any]] = []
    cells: list[npt.NDArray[Any]] = []
    dates: list[npt.NDArray[Any]] = []
    levels: list[npt.NDArray[Any]] = []

    for fire_id, first_day, last_day in zip(
        fires["fire_id"], fires["start_date"], fires["end_date"], strict=True
    ):
        geom = geometry.loc[fire_id]
        for level, covered, from_day, to_day in (
            (LEVEL_POLYGON, spec.polygon_to_cells(geom), first_day, last_day),
            (
                LEVEL_BUFFER,
                spec.polygon_to_cells(geom.buffer(radius)),
                first_day - slack,
                last_day + slack,
            ),
        ):
            if not covered:
                continue
            days = pd.date_range(from_day, to_day).to_numpy()
            size = len(covered) * len(days)
            cells.append(np.repeat(covered, len(days)))
            dates.append(np.tile(days, len(covered)))
            fire_ids.append(np.full(size, fire_id, dtype=np.int64))
            levels.append(np.full(size, level, dtype=np.int8))

    pairs = pd.DataFrame(
        {
            "fire_id": np.concatenate(fire_ids),
            "cell_id": np.concatenate(cells).astype("int32"),
            "date": np.concatenate(dates),
            "level": np.concatenate(levels),
        }
    )
    logger.info(
        "Exclusion pairs: %d from %d fire(s), buffer %d cell(s) and %d day(s)",
        len(pairs),
        len(fires),
        config.sampling.buffer_cells,
        config.sampling.buffer_days,
    )
    return pairs


def unique_exclusions(pairs: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per excluded cell-day, keeping the tightest level that hit it."""
    exclusions = pairs.groupby(["cell_id", "date"], as_index=False).agg(level=("level", "min"))
    logger.info(
        "Excluded cell-days: %d (%s)",
        len(exclusions),
        exclusions["level"].value_counts().sort_index().to_dict(),
    )
    return exclusions


def flag_near_other_fires(
    positives: pd.DataFrame, pairs: pd.DataFrame, fires: pd.DataFrame
) -> npt.NDArray[np.bool_]:
    """True where an event other than the row's own reaches this cell-day.

    A source event is identified by its ignition cell-day, not by `fire_id`, so the
    fires that collapsed into this row do not count as neighbors of themselves.
    """
    source = fires[["fire_id", "cell_id", "start_date"]].rename(
        columns={"cell_id": "source_cell_id", "start_date": "source_date"}
    )
    hits = (
        positives[["cell_id", "date"]]
        .merge(pairs[["fire_id", "cell_id", "date"]], on=["cell_id", "date"])
        .merge(source, on="fire_id")
    )
    other = hits.loc[
        (hits["source_cell_id"] != hits["cell_id"]) | (hits["source_date"] != hits["date"]),
        ["cell_id", "date"],
    ].drop_duplicates()

    near: npt.NDArray[np.bool_] = (
        pd.MultiIndex.from_frame(positives[["cell_id", "date"]])
        .isin(pd.MultiIndex.from_frame(other))
        .astype(bool)
    )
    return near


def _flat_exclusions(
    exclusions: pd.DataFrame, pool: npt.NDArray[Any], days: pd.DatetimeIndex
) -> npt.NDArray[np.int64]:
    """Excluded cell-days as indices into the flattened `pool x days` space.

    Exclusions outside the pool (inactive or non-burnable cells) or outside the date
    range have no index and drop out here.
    """
    position = np.full(int(pool.max()) + 1, -1, dtype=np.int64)
    position[pool] = np.arange(len(pool))

    cell_id = exclusions["cell_id"].to_numpy(dtype=np.int64)
    offset = (exclusions["date"] - days[0]).dt.days.to_numpy()

    within = (cell_id >= 0) & (cell_id < len(position)) & (offset >= 0) & (offset < len(days))
    cell_position = position[cell_id[within]]
    offset = offset[within]

    in_pool = cell_position >= 0
    flat: npt.NDArray[np.int64] = np.unique(cell_position[in_pool] * len(days) + offset[in_pool])
    return flat


def negative_pool(
    grid: pd.DataFrame, exclusions: pd.DataFrame, config: Config
) -> tuple[npt.NDArray[Any], pd.DatetimeIndex, npt.NDArray[np.int64]]:
    """The cell-days a negative can come from: burnable active grid x record, less exclusions.

    Shared with the calibration prior correction, which needs the size of the population the
    negatives were subsampled out of and cannot afford to describe it differently from here.
    """
    pool = grid.loc[grid["is_trentino"] & ~grid["is_non_burnable"], "cell_id"].to_numpy()
    days = pd.date_range(config.date_range.start, config.date_range.end)
    return pool, days, _flat_exclusions(exclusions, pool, days)


def sample_negatives(
    grid: pd.DataFrame,
    exclusions: pd.DataFrame,
    n_negatives: int,
    config: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw cell-days uniformly from the burnable active grid, minus the exclusion set."""
    if config.sampling.hard_negative_fraction > 0:
        raise NotImplementedError(
            "Hard-negative mining draws from high-FWI days, and the FWI series arrives with M5"
        )

    pool, days, blocked = negative_pool(grid, exclusions, config)
    total = len(pool) * len(days)
    available = total - len(blocked)
    if n_negatives > available:
        raise ValueError(f"Asked for {n_negatives} negatives, only {available} cell-days available")

    logger.info(
        "Negative pool: %d cell(s) x %d day(s) = %d, %d excluded (%.3f%%)",
        len(pool),
        len(days),
        total,
        len(blocked),
        100 * len(blocked) / total,
    )

    # drawn in order and truncated at the end, never sorted then cut: the flat index is
    # ordered by cell then day, so keeping the smallest would favor the north-west and
    # the 1980s.
    accepted = np.empty(0, dtype=np.int64)
    while len(accepted) < n_negatives:
        batch = rng.integers(0, total, size=n_negatives - len(accepted))
        batch = batch[~np.isin(batch, blocked) & ~np.isin(batch, accepted)]
        _, first_seen = np.unique(batch, return_index=True)
        accepted = np.concatenate([accepted, batch[np.sort(first_seen)]])

    accepted = accepted[:n_negatives]
    return pd.DataFrame(
        {"cell_id": pool[accepted // len(days)], "date": days[accepted % len(days)]}
    )


def assemble_samples(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Stack the two label classes into the table the feature stages join onto."""
    unlabeled = negatives.assign(
        is_fire=False,
        is_near_fire=False,
        fire_id=pd.Series(pd.NA, index=negatives.index, dtype="Int64"),
        n_fires=0,
    )
    samples = pd.concat([positives.assign(is_fire=True), unlabeled], ignore_index=True).sort_values(
        ["date", "cell_id"], kind="stable", ignore_index=True
    )

    samples.insert(0, "sample_id", np.arange(len(samples), dtype="int32"))
    return samples[
        ["sample_id", "cell_id", "date", "is_fire", "is_near_fire", "fire_id", "n_fires"]
    ].astype(
        {
            "cell_id": "int32",
            "is_fire": bool,
            "is_near_fire": bool,
            "fire_id": "Int64",
            "n_fires": "int16",
        }
    )


def validate_samples(samples: pd.DataFrame, exclusions: pd.DataFrame, config: Config) -> None:
    """Check the acceptance criteria, logging loudly."""
    positives = samples.loc[samples["is_fire"]]
    negatives = samples.loc[~samples["is_fire"]]

    expected = config.sampling.expected_positives
    if len(positives) != expected:
        logger.error("Positives: %d, expected %d", len(positives), expected)
    else:
        logger.info(
            "Positives: %d (as expected), covering %d fire(s)",
            len(positives),
            int(positives["n_fires"].sum()),
        )

    ratio = config.sampling.negative_ratio
    if len(negatives) != ratio * len(positives):
        logger.error("Negatives: %d, expected %d", len(negatives), ratio * len(positives))
    else:
        logger.info("Negatives: %d at ratio 1:%d", len(negatives), ratio)

    leaked = int(
        pd.MultiIndex.from_frame(negatives[["cell_id", "date"]])
        .isin(pd.MultiIndex.from_frame(exclusions[["cell_id", "date"]]))
        .sum()
    )
    if leaked:
        logger.error("%d sampled negative(s) fall inside the exclusion set", leaked)

    duplicated = int(samples.duplicated(subset=["cell_id", "date"]).sum())
    if duplicated:
        logger.error("%d duplicated (cell_id, date) pair(s)", duplicated)

    logger.info(
        "Positives also reached by another event (is_near_fire): %d",
        int(positives["is_near_fire"].sum()),
    )
    logger.info(
        "Sample dates: %s to %s", samples["date"].min().date(), samples["date"].max().date()
    )


def build_samples(config: Config, force: bool = False) -> pd.DataFrame:
    """Build `samples.parquet` and the companion `exclusions.parquet`."""
    samples_out = config.path(config.paths.samples_out)
    exclusions_out = config.path(config.paths.exclusions_out)

    if samples_out.exists() and exclusions_out.exists() and not force:
        logger.info("Outputs already exist, skipping (use --force to rebuild): %s", samples_out)
        return pd.read_parquet(samples_out)

    spec, grid = load_grid(config)
    fires = prepare_fires(pd.read_parquet(config.path(config.paths.fires_out)), spec)
    polygons = gpd.read_parquet(config.path(config.paths.fire_polygons_out))

    positives = build_positive_rows(fires, grid, config)

    # every dated fire contributes exclusions, including the ones dropped as positives:
    # a burn still happened there, whatever the label table ended up holding.
    pairs = exclusion_pairs(fires, polygons, spec, config)
    exclusions = unique_exclusions(pairs)
    positives["is_near_fire"] = flag_near_other_fires(positives, pairs, fires)

    rng = np.random.default_rng(config.project.random_seed)
    negatives = sample_negatives(
        grid, exclusions, config.sampling.negative_ratio * len(positives), config, rng
    )

    samples = assemble_samples(positives, negatives)
    validate_samples(samples, exclusions, config)

    samples_out.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(samples_out, index=False)
    exclusions.to_parquet(exclusions_out, index=False)
    logger.info("Wrote %d rows to %s", len(samples), samples_out)
    logger.info("Wrote %d excluded cell-days to %s", len(exclusions), exclusions_out)

    return samples
