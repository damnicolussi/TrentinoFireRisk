"""Curves, reliability, per-fold metrics, attribution and sensitivity, written as PNGs."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tfire.config import Config

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tfire.models.explain import Attribution

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots in their fixed order; a model keeps its color across every figure
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
# same hues carry the two-way splits (static/dynamic, raw/calibrated)
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
SEQUENTIAL = "#2a78d6"

# line style as well as hue, so a curve stays identifiable in grayscale
DASHES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)))

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "xgboost_no_stacking": "XGBoost, no stacking",
    "random_forest": "Random forest",
    "logistic": "Logistic regression",
    "fwi_only": "FWI only",
}

_RC: dict[Any, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.size": 9,
    "lines.linewidth": 1.6,
}


def _style() -> AbstractContextManager[None]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt.rc_context(_RC)


def _save(figure: Figure, config: Config, name: str) -> Path:
    from matplotlib import pyplot as plt

    directory = config.path(config.paths.figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=config.evaluation.figure_dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _label(axes: Axes, title: str, x: str, y: str) -> None:
    axes.set_title(title, loc="left", pad=10)
    axes.set_xlabel(x)
    axes.set_ylabel(y)


def curves(
    labels: npt.NDArray[Any],
    probabilities: dict[str, npt.NDArray[Any]],
    config: Config,
) -> list[Path]:
    """Precision-recall and ROC over the holdout years, the model against its baselines."""
    from matplotlib import pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    base_rate = float(labels.mean())
    written = []
    with _style():
        precision_figure, precision_axes = plt.subplots(figsize=(5.5, 4.2))
        roc_figure, roc_axes = plt.subplots(figsize=(5.5, 4.2))

        for index, (name, scores) in enumerate(probabilities.items()):
            style: dict[str, Any] = {
                "color": SERIES[index % len(SERIES)],
                "linestyle": DASHES[index % len(DASHES)],
            }
            label = MODEL_LABELS.get(name, name)

            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = average_precision_score(labels, scores)
            precision_axes.plot(recall, precision, label=f"{label}  {auprc:.3f}", **style)

            false_rate, true_rate, _ = roc_curve(labels, scores)
            auroc = roc_auc_score(labels, scores)
            roc_axes.plot(false_rate, true_rate, label=f"{label}  {auroc:.3f}", **style)

        precision_axes.axhline(base_rate, color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)))
        precision_axes.annotate(
            f"base rate {base_rate:.1%}",
            xy=(0.02, base_rate),
            xytext=(0, 4),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=SECONDARY_INK,
        )
        _label(precision_axes, "Precision-recall, holdout 2015-2024 (AUPRC)", "Recall", "Precision")
        precision_axes.set_ylim(0, 1)
        precision_axes.legend(loc="upper right")

        roc_axes.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)))
        _label(
            roc_axes,
            "ROC, holdout 2015-2024 (AUROC)",
            "False positive rate",
            "True positive rate",
        )
        roc_axes.legend(loc="lower right")

        written.append(_save(precision_figure, config, "pr_curves"))
        written.append(_save(roc_figure, config, "roc_curves"))
    return written


def reliability(calibration: dict[str, Any], config: Config) -> Path:
    """Predicted against observed frequency, before and after the isotonic fit."""
    from matplotlib import pyplot as plt

    with _style():
        figure, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(5.0, 5.2),
            height_ratios=(3, 1),
            sharex=True,
            constrained_layout=True,
        )
        top.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)))

        for index, name in enumerate(("raw", "isotonic")):
            frame = pd.DataFrame(calibration[name]["bins"])
            top.plot(
                frame["predicted"],
                frame["observed"],
                marker="o",
                markersize=4,
                color=SERIES[index],
                linestyle=DASHES[index],
                label=f"{name}  ECE {calibration[name]['ece']:.3f}",
            )
            bottom.step(
                frame["predicted"], frame["count"], where="mid", color=SERIES[index], linewidth=1.2
            )

        # both axes on the same range, so the diagonal stays at 45 degrees and the gap to it
        # is the miscalibration rather than an artifact of the aspect
        limit = min(
            1.0,
            1.1
            * max(
                float(pd.DataFrame(calibration[name]["bins"])[axis].max())
                for name in ("raw", "isotonic")
                for axis in ("predicted", "observed")
            ),
        )
        top.set_xlim(0, limit)
        top.set_ylim(0, limit)
        _label(top, "Reliability on the holdout years", "", "Observed frequency")
        top.legend(loc="upper left")
        _label(bottom, "", "Predicted probability", "Rows in bin")
        bottom.set_yscale("log")

        return _save(figure, config, "reliability")


def fold_metrics(folds: list[dict[str, Any]], config: Config) -> Path:
    """AUPRC, AUROC and base rate per year block, the three of them on one scale."""
    from matplotlib import pyplot as plt

    frame = pd.DataFrame(folds)
    position = np.arange(len(frame))

    with _style():
        figure, axes = plt.subplots(figsize=(5.5, 3.8))
        for index, (column, label) in enumerate(
            (("auprc", "AUPRC"), ("auroc", "AUROC"), ("positive_rate", "Base rate"))
        ):
            axes.plot(
                position,
                frame[column],
                marker="o",
                markersize=5,
                color=SERIES[index],
                linestyle=DASHES[index],
                label=label,
            )
            for x, value in zip(position, frame[column], strict=True):
                axes.annotate(
                    f"{value:.2f}",
                    xy=(x, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=SECONDARY_INK,
                )

        axes.set_xticks(position, frame["years"])
        axes.set_ylim(0, 1)
        _label(axes, "Out-of-fold performance per year block", "", "")
        axes.legend(loc="center left")
        return _save(figure, config, "fold_metrics")


def attribution(explained: Attribution, config: Config) -> list[Path]:
    """The beeswarm and the per-category totals."""
    import shap
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    written = []
    with _style():
        shap.summary_plot(
            explained.values,
            explained.features,
            max_display=config.evaluation.shap_max_display,
            show=False,
            plot_size=(6.5, 6.0),
        )
        figure = plt.gcf()
        figure.axes[0].set_title("Feature attribution on the holdout years", loc="left", pad=10)
        written.append(_save(figure, config, "shap_beeswarm"))

        frame = explained.per_category.sort_values("mean_abs_shap")
        colors = {"static": SERIES[0], "dynamic": SERIES[1], "unknown": MUTED}
        temporal = (
            explained.per_feature.groupby("category")["temporal"].agg(
                lambda values: values.mode().iat[0]
            )
        ).to_dict()

        figure, axes = plt.subplots(figsize=(5.5, 3.8))
        axes.barh(
            frame["category"],
            frame["share"],
            color=[colors.get(temporal.get(name, "unknown"), MUTED) for name in frame["category"]],
            height=0.68,
        )
        for y, (value, count) in enumerate(zip(frame["share"], frame["features"], strict=True)):
            axes.annotate(
                f"{value:.1%}  ({count})",
                xy=(value, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=SECONDARY_INK,
            )

        axes.set_xlim(0, float(frame["share"].max()) * 1.25)
        axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axes.grid(axis="y", visible=False)
        static = _temporal_share(explained, "static")
        _label(
            axes,
            f"Share of |SHAP| per category: {static:.0%} static, {1 - static:.0%} dynamic",
            "Share of total mean |SHAP|",
            "",
        )
        handles = [Rectangle((0, 0), 1, 1, color=colors[name]) for name in ("static", "dynamic")]
        axes.legend(handles, ("static", "dynamic"), loc="lower right")
        written.append(_save(figure, config, "shap_categories"))
    return written


def _temporal_share(explained: Attribution, key: str) -> float:
    match = explained.per_temporal.loc[explained.per_temporal["temporal"] == key, "share"]
    return float(match.iloc[0]) if len(match) else 0.0


def sensitivity(results: dict[str, Any], config: Config) -> Path | None:
    """How far each disturbed assumption moves the holdout scores.

    Both metrics, because they answer different questions here: changing the negative ratio
    changes the base rate, and AUPRC is defined against it while AUROC is not.
    """
    from matplotlib import pyplot as plt

    if not results:
        return None

    frame = pd.DataFrame(
        {
            "variant": list(results),
            "auprc": [item["delta"]["holdout_auprc"] for item in results.values()],
            "auroc": [item["delta"]["holdout_auroc"] for item in results.values()],
        }
    ).sort_values("auprc", ignore_index=True)

    position = np.arange(len(frame))
    height = 0.32
    with _style():
        figure, axes = plt.subplots(figsize=(5.8, 0.85 * len(frame) + 1.6))
        for index, (column, label) in enumerate((("auprc", "AUPRC"), ("auroc", "AUROC"))):
            offset = (0.5 - index) * height * 1.08
            axes.barh(
                position + offset,
                frame[column],
                height=height,
                color=SERIES[index],
                label=label,
            )
            for y, value in zip(position + offset, frame[column], strict=True):
                axes.annotate(
                    f"{value:+.4f}",
                    xy=(value, y),
                    xytext=(6 if value >= 0 else -6, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=7,
                    color=SECONDARY_INK,
                )

        axes.axvline(0, color=AXIS, linewidth=1.0)
        axes.set_yticks(position, frame["variant"])
        span = max(float(frame[["auprc", "auroc"]].abs().to_numpy().max()) * 1.7, 0.01)
        axes.set_xlim(-span, span)
        axes.grid(axis="y", visible=False)
        _label(axes, "Holdout scores against the canonical run", "Difference", "")
        axes.legend(loc="lower right")
        return _save(figure, config, "sensitivity")


def spatial_comparison(temporal: dict[str, Any], spatial: dict[str, Any], config: Config) -> Path:
    """The same rows and the same model, partitioned by year and by spatial block."""
    from matplotlib import pyplot as plt

    metrics = ("auprc", "auroc")
    position = np.arange(len(metrics))
    width = 0.24

    blocks = f"{config.evaluation.spatial_block_m / 1000:.0f} km blocks"
    schemes = (("year blocks", temporal), (blocks, spatial))
    with _style():
        figure, axes = plt.subplots(figsize=(4.6, 3.4))
        for index, (name, block) in enumerate(schemes):
            values = [block[metric] for metric in metrics]
            offset = (index - 0.5) * width * 1.08
            axes.bar(position + offset, values, width=width, color=SERIES[index], label=name)
            for x, value in zip(position + offset, values, strict=True):
                axes.annotate(
                    f"{value:.3f}",
                    xy=(x, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=SECONDARY_INK,
                )

        axes.set_xticks(position, [metric.upper() for metric in metrics])
        axes.set_ylim(0, 1)
        axes.grid(axis="x", visible=False)
        _label(axes, "Pooled out-of-fold score by blocking scheme", "", "")
        axes.legend(loc="upper left")
        return _save(figure, config, "blocking_schemes")
