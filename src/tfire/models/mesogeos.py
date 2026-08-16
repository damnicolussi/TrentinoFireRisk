"""Stage-one fire danger model: XGBoost trained on Mesogeos, applied to Trentino feature values."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config
from tfire.evaluation import scores
from tfire.sources.dem import N_ASPECT_CLASSES, aspect_sectors

if TYPE_CHECKING:
    from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_FILENAME: Final = "model.json"
METRICS_FILENAME: Final = "metrics.json"
MAPPING_FILENAME: Final = "mapping.json"

PROB_COLUMN: Final = "mesogeos_prob"

_TRACKS: Final[dict[str, int]] = {"positives": 1, "negatives": 0}

LAND_COVER_CLASSES: Final = (
    "agriculture",
    "forest",
    "grassland",
    "settlement",
    "shrubland",
    "sparse_vegetation",
    "water_bodies",
    "wetland",
)

PRECIP_WINDOWS: Final = (7, 15, 30)

# CORINE grid codes 1-44, each landing in exactly one Mesogeos class so the eight fractions
# still sum to 1. Code 18 is CLC 231 pastures: managed grass, grouped with natural grassland
# rather than with the arable classes it sits beside in the CORINE legend.
CLC_TO_MESOGEOS: Final[dict[str, tuple[int, ...]]] = {
    "settlement": tuple(range(1, 12)),
    "agriculture": (12, 13, 14, 15, 16, 17, 19, 20, 21, 22),
    "grassland": (18, 26),
    "forest": (23, 24, 25),
    "shrubland": (27, 28, 29),
    "sparse_vegetation": (30, 31, 32, 33, 34),
    "wetland": (35, 36, 37, 38, 39),
    "water_bodies": (40, 41, 42, 43, 44),
}

# canonical name -> (Mesogeos column, multiplier, offset). Mesogeos ships kelvin, pascals,
# metres of rain, kilometres of distance and humidity as a fraction.
MESOGEOS_SCALING: Final[dict[str, tuple[str, float, float]]] = {
    "elevation_m": ("dem", 1.0, 0.0),
    "ndvi": ("ndvi", 1.0, 0.0),
    "temp_max_c": ("t2m", 1.0, -273.15),
    "rh_min_pct": ("rh", 100.0, 0.0),
    "wind_speed_max_ms": ("wind_speed", 1.0, 0.0),
    "pres_max_hpa": ("sp", 0.01, 0.0),
    "precip_mm": ("tp", 1000.0, 0.0),
    "dist_roads_m": ("roads_distance", 1000.0, 0.0),
    "pop_density_km2": ("population", 1.0, 0.0),
}

# Mesogeos aggregates each day to an extreme, not to a mean, so the Trentino counterparts are
# the daily extremes too.
TRENTINO_COLUMNS: Final[dict[str, str]] = {
    "elevation_m": "elevation_mean",
    "ndvi": "ndvi",
    "temp_max_c": "temp_max",
    "rh_min_pct": "rh_min",
    "wind_speed_max_ms": "wind_speed_max",
    "pres_max_hpa": "pres_max",
    "precip_mm": "precip_sum",
    **{f"precip_cum{window}_mm": f"precip_cum{window}" for window in PRECIP_WINDOWS},
    "dist_roads_m": "dist_roads_mean",
    "pop_density_km2": "pop_density",
}

COMMON_FEATURES: Final = (
    "elevation_m",
    "northness",
    "eastness",
    "ndvi",
    "temp_max_c",
    "rh_min_pct",
    "wind_speed_max_ms",
    "pres_max_hpa",
    "precip_mm",
    *(f"precip_cum{window}_mm" for window in PRECIP_WINDOWS),
    "dist_roads_m",
    "pop_density_km2",
    *(f"lc_{name}" for name in LAND_COVER_CLASSES),
)

UNMAPPED: Final[dict[str, str]] = {
    "slope": (
        "documented as radians but the median is 1.5498 rad (88.8 degrees) and the lower "
        "quartile is already 1.523: a saturated upstream computation, not comparable to "
        "slope_mean in degrees"
    ),
    "curvature": "no Trentino counterpart",
    "lst_day": (
        "MODIS daily land surface temperature against a Landsat monthly composite, and "
        "missing on 28% of the Mesogeos target days"
    ),
    "lst_night": "no Trentino counterpart, the Landsat composite is a daytime overpass",
    "wind_direction": (
        "Mediterranean circulation patterns carry no information about alpine valley winds"
    ),
    "lai": "no Trentino counterpart",
    "smi": "no Trentino counterpart",
    "ssrd": "no Trentino counterpart",
    "d2m": "no Trentino counterpart, humidity enters through rh instead",
    "x": "would let the base model memorize Mediterranean geography",
    "y": "would let the base model memorize Mediterranean geography",
    "burned_areas": "label",
    "ignition_points": "label",
    "burned_area_has": "label",
}


def _derivation(name: str) -> tuple[str, str]:
    """Where a common feature comes from on each side, for the features that are not a rename."""
    if name in ("northness", "eastness"):
        return "aspect", "aspect_1 ... aspect_8"
    if name.startswith("precip_cum"):
        return "tp, trailing sum over the track", TRENTINO_COLUMNS[name]
    if name.startswith("lc_"):
        codes = CLC_TO_MESOGEOS[name.removeprefix("lc_")]
        return name, " + ".join(f"CLC_{code}" for code in codes)
    return MESOGEOS_SCALING[name][0], TRENTINO_COLUMNS[name]


def mapping_table() -> dict[str, list[dict[str, str]]]:
    """The mapped and unmapped Mesogeos variables, the milestone's documented deliverable."""
    mapped = []
    for name in COMMON_FEATURES:
        source, target = _derivation(name)
        mapped.append({"common": name, "mesogeos": source, "trentino": target})
    return {
        "mapped": mapped,
        "unmapped": [{"mesogeos": name, "reason": reason} for name, reason in UNMAPPED.items()],
    }


def sector_centers() -> list[float]:
    """Mid-bearing of each aspect sector, in degrees.

    Derived from the sectors rather than recomputed, so a change to the sector layout cannot
    leave the centers behind. The sector crossing north has its end below its start.
    """
    return [
        (start + end) / 2 if start < end else ((start + end + 360.0) / 2) % 360.0
        for start, end in aspect_sectors()
    ]


def aspect_components(
    frame: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Northness and eastness from the eight aspect fractions.

    `aspect_NODATA` is left out of the sum, so a flat cell yields (0, 0) instead of pointing
    somewhere arbitrary, and a cell split evenly between opposing slopes cancels the same way.
    """
    centers = np.deg2rad(sector_centers())
    columns = [f"aspect_{index + 1}" for index in range(N_ASPECT_CLASSES)]
    weights = frame[columns].to_numpy("float64")
    return weights @ np.cos(centers), weights @ np.sin(centers)


def _land_cover_from_clc(frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
    return {
        f"lc_{name}": frame[[f"CLC_{code}" for code in codes]].to_numpy("float64").sum(axis=1)
        for name, codes in CLC_TO_MESOGEOS.items()
    }


def from_trentino(frame: pd.DataFrame) -> pd.DataFrame:
    """The common feature vector, read off the assembled training table."""
    northness, eastness = aspect_components(frame)
    columns: dict[str, npt.NDArray[np.float64]] = {
        canonical: frame[source].to_numpy("float64")
        for canonical, source in TRENTINO_COLUMNS.items()
    }
    columns["northness"] = northness
    columns["eastness"] = eastness
    columns.update(_land_cover_from_clc(frame))
    return pd.DataFrame(columns, index=frame.index)[list(COMMON_FEATURES)]


def _read_track(path: Path, label: int, config: Config) -> pd.DataFrame:
    """One track file as target-day rows, with the cumulative precipitation windows attached."""
    needed = {"time", "time_idx", "sample", "aspect", "tp"}
    needed |= {f"lc_{name}" for name in LAND_COVER_CLASSES}
    needed |= {source for source, _, _ in MESOGEOS_SCALING.values()}
    raw = pd.read_csv(path, usecols=sorted(needed), parse_dates=["time"])

    # `sample` restarts from 0 in each file, so the two sets of ids overlap completely
    raw["track_id"] = f"{path.stem}-" + raw["sample"].astype(str)
    ordered = raw.sort_values(["track_id", "time_idx"], kind="stable")

    length = config.mesogeos.track_length_days
    sizes = ordered.groupby("track_id", sort=False).size()
    ragged = int((sizes != length).sum())
    if ragged:
        raise ValueError(f"{path.name}: {ragged} track(s) are not {length} days long")

    daily_precip = ordered["tp"].to_numpy("float64").reshape(-1, length) * 1000.0
    target = ordered.loc[ordered["time_idx"] == config.mesogeos.target_time_idx].reset_index(
        drop=True
    )
    if len(target) != len(daily_precip):
        raise ValueError(f"{path.name}: {len(target)} target days for {len(daily_precip)} tracks")

    columns: dict[str, npt.NDArray[np.float64]] = {
        canonical: target[source].to_numpy("float64") * scale + offset
        for canonical, (source, scale, offset) in MESOGEOS_SCALING.items()
    }
    bearing = np.deg2rad(target["aspect"].to_numpy("float64"))
    columns["northness"] = np.cos(bearing)
    columns["eastness"] = np.sin(bearing)
    for window in PRECIP_WINDOWS:
        columns[f"precip_cum{window}_mm"] = daily_precip[:, -window:].sum(axis=1)
    for name in LAND_COVER_CLASSES:
        columns[f"lc_{name}"] = target[f"lc_{name}"].to_numpy("float64")

    frame = pd.DataFrame(columns)[list(COMMON_FEATURES)]
    frame["label"] = label
    frame["year"] = target["time"].dt.year.to_numpy()
    return frame


def load_mesogeos(config: Config) -> pd.DataFrame:
    """Both track files as one table of target days: the 22 common features, label and year."""
    directory = config.path(config.paths.mesogeos_raw)
    frames = []
    for stem, label in _TRACKS.items():
        path = directory / f"{stem}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Mesogeos track file not found: {path}")
        frames.append(_read_track(path, label, config))

    frame = pd.concat(frames, ignore_index=True)
    logger.info(
        "Mesogeos: %d target days, %d positive, %d to %d",
        len(frame),
        int(frame["label"].sum()),
        int(frame["year"].min()),
        int(frame["year"].max()),
    )
    return frame


def _classifier(config: Config, n_estimators: int, early_stopping: int | None) -> XGBClassifier:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=config.mesogeos.max_depth,
        learning_rate=config.mesogeos.learning_rate,
        early_stopping_rounds=early_stopping,
        eval_metric="aucpr",
        random_state=config.project.random_seed,
        n_jobs=-1,
    )


def train_mesogeos(config: Config, force: bool = False) -> dict[str, Any]:
    """Fit the base model on the Mesogeos tracks and persist it under `models/mesogeos/`."""
    directory = config.path(config.paths.mesogeos_model_dir)
    model_path = directory / MODEL_FILENAME
    if model_path.exists() and not force:
        logger.info("Model already exists, skipping (use --force to retrain): %s", model_path)
        stored: dict[str, Any] = json.loads(
            (directory / METRICS_FILENAME).read_text(encoding="utf-8")
        )
        return stored

    frame = load_mesogeos(config)
    features = list(COMMON_FEATURES)
    holdout = frame["year"].isin(config.mesogeos.holdout_years)
    if not holdout.any() or holdout.all():
        raise ValueError(f"holdout_years {config.mesogeos.holdout_years} splits nothing off")

    train, test = frame.loc[~holdout], frame.loc[holdout]
    positives = int(train["label"].sum())
    weight = (len(train) - positives) / positives

    tuned = _classifier(config, config.mesogeos.n_estimators, config.mesogeos.early_stopping_rounds)
    tuned.set_params(scale_pos_weight=weight)
    tuned.fit(
        train[features],
        train["label"],
        eval_set=[(test[features], test["label"])],
        verbose=False,
    )
    rounds = int(tuned.best_iteration) + 1
    measured = scores(test["label"].to_numpy(), tuned.predict_proba(test[features])[:, 1])
    logger.info(
        "Holdout %s: AUPRC %.3f, AUROC %.3f against a %.1f%% base rate, %d rounds",
        config.mesogeos.holdout_years,
        measured["auprc"],
        measured["auroc"],
        100 * measured["positive_rate"],
        rounds,
    )

    # the holdout years carry a fifth of the record; the shipped model sees them too, at the
    # round count the holdout settled on
    positives = int(frame["label"].sum())
    final = _classifier(config, rounds, None)
    final.set_params(scale_pos_weight=(len(frame) - positives) / positives)
    final.fit(frame[features], frame["label"], verbose=False)

    metrics = {
        "holdout_years": config.mesogeos.holdout_years,
        "holdout": measured,
        "rounds": rounds,
        "rows": {"train": len(train), "holdout": len(test), "final_fit": len(frame)},
        "positives": {"train": int(train["label"].sum()), "final_fit": positives},
        "scale_pos_weight": (len(frame) - positives) / positives,
        "hyperparameters": {
            "max_depth": config.mesogeos.max_depth,
            "learning_rate": config.mesogeos.learning_rate,
            "n_estimators": config.mesogeos.n_estimators,
            "early_stopping_rounds": config.mesogeos.early_stopping_rounds,
        },
        "random_seed": config.project.random_seed,
        "features": features,
    }
    directory.mkdir(parents=True, exist_ok=True)
    final.save_model(model_path)
    (directory / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (directory / MAPPING_FILENAME).write_text(
        json.dumps(mapping_table(), indent=2), encoding="utf-8"
    )
    logger.info("Wrote the base model and its %d mapped features to %s", len(features), directory)
    return metrics


def predict_prob(frame: pd.DataFrame, config: Config) -> npt.NDArray[np.float32]:
    """Base-model probability for each row of the assembled Trentino table."""
    from xgboost import XGBClassifier

    model_path = config.path(config.paths.mesogeos_model_dir) / MODEL_FILENAME
    model = XGBClassifier()
    model.load_model(model_path)

    stored = model.get_booster().feature_names
    if stored != list(COMMON_FEATURES):
        raise ValueError(
            f"Stored model expects {stored}, the common schema is {list(COMMON_FEATURES)}. "
            "Retrain with `tfire train-mesogeos --force`."
        )

    scores = model.predict_proba(from_trentino(frame))
    probabilities: npt.NDArray[np.float32] = scores[:, 1].astype("float32")
    return probabilities
