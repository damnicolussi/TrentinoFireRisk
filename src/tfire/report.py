"""Render the evaluation numbers as markdown."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tfire.config import Config
from tfire.figures import MODEL_LABELS

logger = logging.getLogger(__name__)

REPORT_FILENAME = "evaluation.md"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [f"| {' | '.join(headers)} |", f"|{'---|' * len(headers)}"]
    lines += [f"| {' | '.join(str(cell) for cell in row)} |" for row in rows]
    return lines


def _model_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            MODEL_LABELS.get(name, name),
            f"{block['pooled_out_of_fold']['auprc']:.4f}",
            f"{block['pooled_out_of_fold']['auroc']:.4f}",
            f"{block['pooled_out_of_fold']['lift']:.2f}",
            f"{block['holdout']['auprc']:.4f}",
            f"{block['holdout']['auroc']:.4f}",
            f"{block['holdout']['lift']:.2f}",
        ]
        for name, block in metrics["models"].items()
    ]


def _sensitivity_rows(results: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            name,
            block["rows"],
            block["positives"],
            f"{block['holdout']['positive_rate']:.2%}",
            f"{block['holdout']['auprc']:.4f}",
            f"{block['delta']['holdout_auprc']:+.4f}",
            f"{block['holdout']['auroc']:.4f}",
            f"{block['delta']['holdout_auroc']:+.4f}",
        ]
        for name, block in results.items()
    ]


def _widest(results: dict[str, Any], metric: str) -> float:
    return max(abs(float(block["delta"][f"holdout_{metric}"])) for block in results.values())


def _event_rows(verification: dict[str, Any]) -> list[list[Any]]:
    baseline = verification.get("baseline")
    blocks = [("all", verification["overall"]), *sorted(verification["by_season"].items())]
    rows = []
    for name, block in blocks:
        row = [
            name,
            block["events"],
            f"{block['median_percentile']:.3f}",
            f"{block['share_at_or_above_90']:.1%}",
            f"{block['share_at_or_above_99']:.1%}",
        ]
        if baseline:
            other = baseline["overall"] if name == "all" else baseline["by_season"][name]
            row += [
                f"{other['median_percentile']:.3f}",
                f"{other['share_at_or_above_90']:.1%}",
                f"{other['share_at_or_above_99']:.1%}",
            ]
        rows.append(row)
    return rows


def _event_section(verification: dict[str, Any]) -> list[str]:
    years = verification["years"]
    share = verification["cell_effect_variance_share"]
    headers = ["events", "n", "median percentile", "at or above 90th", "at or above 99th"]
    intro = [
        f"Every cadastre ignition of {years[0]}-{years[1]} against the map its own day was "
        "scored on, ranked within that day.",
    ]
    baseline = verification.get("baseline")
    if baseline:
        headers += ["baseline median", "baseline 90th", "baseline 99th"]
        intro += [
            f"The map is drawn by the {verification['model']}, so none of these ignitions was "
            f"seen in fitting. The baseline is the {baseline['name']}, ranked over the same "
            "grid: what knowing only where fires started before would give.",
        ]
    return [
        "## Where recorded ignitions landed",
        "",
        " ".join(intro),
        "",
        *table(headers, _event_rows(verification)),
        "",
        *table(
            ("class", "ignition cells"),
            [[key, value] for key, value in verification["class_distribution"].items()],
        ),
        "",
        f"Which cell a value belongs to explains {share:.1%} of the variance in log10(p) over "
        f"{len(verification['variance_days'])} holdout days spread across the record, and "
        f"{verification['cell_effect_variance_share_in_season']:.1%} over the "
        f"{len(verification['season_window'])} consecutive August days at the end of it. The "
        "second is the number an operator meets: inside one fire season the weather moves "
        "little, so whatever is fixed about a cell is most of what separates two of them.",
        "",
    ]


def _window_section(window: dict[str, Any] | None) -> list[str]:
    if not window:
        return []
    served = window["served"]
    first, last = served["window"]["years"]
    after = window["validation"][0]["observed_after"]
    observed = after["rate"]
    rows = [
        [
            f"{row['window'][0]}-{row['window'][1]}",
            f"{row['window_rate']:.2e}",
            f"{row['prior_shift']:.3f}",
            f"{row['predicted_after_prior']:.2e} ({row['predicted_after_prior'] / observed:.2f})",
            f"{row['empirical_shift']:.3f}",
            f"{row['predicted_after_empirical']:.2e} "
            f"({row['predicted_after_empirical'] / observed:.2f})",
            row["chosen"],
            "yes" if row["inside_interval"] else "no",
        ]
        for row in window["validation"]
    ]
    return [
        "### Shift to the current rate",
        "",
        f"The offset above lands the probabilities on the average rate of the record, "
        f"{served['fitted_on']['rate']:.2e} per cell-day. The map serves the rate of "
        f"{first}-{last} instead, {served['window']['rate']:.2e} ({served['window']['ignitions']} "
        f"ignitions): a log-odds shift of {served['prior_shift']:.3f} by the prior correction and "
        f"{served['empirical_shift']:.3f} by matching the whole-grid mean over "
        f"{served['days']} sampled days. The {served['chosen']} one is used "
        f"({served['reason']}). Classes and rankings do not move; probabilities and return "
        f"periods do, by a factor of {math.exp(-served['shift']):.2f}.",
        "",
        f"Whether a window predicts the decade after it: the shift fitted on each window before "
        f"{after['years'][0]} with the model fitted on the training years alone, against the "
        f"{after['ignitions']} ignitions of {after['years'][0]}-{after['years'][1]} "
        f"({after['rate']:.2e}, 95% interval {after['low']:.2e}-{after['high']:.2e}). Without "
        f"any shift that model predicts {window['validation'][0]['unshifted_after']:.2e}. The rule "
        "picks the empirical shift when the prior one misses the window's own interval.",
        "",
        *table(
            (
                "window",
                "window rate",
                "prior shift",
                "predicted after (x observed)",
                "empirical shift",
                "predicted after (x observed)",
                "rule picks",
                "pick inside interval",
            ),
            rows,
        ),
        "",
    ]


def render_report(
    evaluation: dict[str, Any],
    metrics: dict[str, Any],
    figure_paths: Sequence[Path],
    config: Config,
) -> Path:
    """Write `reports/evaluation.md` from the two JSON artifacts and the figures on disk."""
    root = config.path(config.paths.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = root / REPORT_FILENAME

    split = metrics["split"]
    calibration = evaluation["calibration"]
    correction = evaluation["sampling_correction"]
    attribution = evaluation["attribution"]

    lines = [
        f"# Evaluation, model {evaluation['version']}",
        "",
        f"Train {split['train_years'][0]}-{split['train_years'][1]}, "
        f"{evaluation['rows']['train']} rows. "
        f"Holdout {split['test_years'][0]}-{split['test_years'][1]}, "
        f"{evaluation['rows']['holdout']} rows, never seen in tuning.",
        "",
        "## Models",
        "",
        *table(
            ("model", "OOF AUPRC", "OOF AUROC", "OOF lift", "holdout AUPRC", "AUROC", "lift"),
            _model_rows(metrics),
        ),
        "",
        f"Base rate {evaluation['pooled_out_of_fold']['positive_rate']:.2%} out of fold and "
        f"{evaluation['holdout']['positive_rate']:.2%} on the holdout. AUPRC moves with the base "
        "rate by construction, so `lift` is the column that compares two splits.",
        "",
        "## Per year block",
        "",
        *table(
            ("years", "rows", "AUPRC", "AUROC", "base rate", "lift"),
            [
                [
                    fold["years"],
                    fold["rows"],
                    f"{fold['auprc']:.4f}",
                    f"{fold['auroc']:.4f}",
                    f"{fold['positive_rate']:.2%}",
                    f"{fold['lift']:.2f}",
                ]
                for fold in evaluation["folds"]
            ],
        ),
        "",
        "## Where the score comes from",
        "",
        f"Partitioning the same rows by {evaluation['spatial_cv']['block_m'] / 1000:.0f} km block "
        "instead of by year gives pooled AUPRC "
        f"{evaluation['spatial_cv']['pooled_out_of_fold']['auprc']:.4f} against "
        f"{evaluation['pooled_out_of_fold']['auprc']:.4f}, AUROC "
        f"{evaluation['spatial_cv']['pooled_out_of_fold']['auroc']:.4f} against "
        f"{evaluation['pooled_out_of_fold']['auroc']:.4f}.",
        "",
        *table(
            ("category", "SHAP share", "features"),
            [
                [row["category"], f"{row['share']:.1%}", row["features"]]
                for row in attribution["per_category"]
            ],
        ),
        "",
        *table(
            ("temporal", "SHAP share", "features"),
            [
                [row["temporal"], f"{row['share']:.1%}", row["features"]]
                for row in attribution["per_temporal"]
            ],
        ),
        "",
        "## Precision at the top of the ranking",
        "",
        *table(
            ("top", "rows", "caught", "precision", "recall", "threshold"),
            [
                [
                    f"{row['fraction']:.0%}",
                    row["k"],
                    row["caught"],
                    f"{row['precision']:.3f}",
                    f"{row['recall']:.3f}",
                    f"{row['threshold']:.4f}",
                ]
                for row in evaluation["precision_at_k"]["holdout"]
            ],
        ),
        "",
        "## Calibration",
        "",
        *table(
            ("probabilities", "ECE", "Brier", "mean predicted"),
            [
                [
                    name,
                    f"{calibration[name]['ece']:.4f}",
                    f"{calibration[name]['brier']:.4f}",
                    f"{calibration[name]['mean_predicted']:.4f}",
                ]
                for name in ("raw", "isotonic")
            ],
        ),
        "",
        f"Negatives were drawn at 1 in {1 / correction['sampling_rate']:.0f} of the "
        f"{correction['population_negatives']:,} cell-days in the pool, so a log-odds offset of "
        f"{correction['log_offset']:.3f} carries a sample-relative probability to the rate of a "
        "cell burning on a given day. Mean predicted rate on the holdout after both steps: "
        f"{calibration['population']['mean_predicted']:.2e}.",
        "",
        *_window_section(evaluation.get("calibration_window")),
    ]

    intervals = evaluation.get("holdout_intervals") or {}
    if intervals:
        lines += [
            "## Holdout intervals",
            "",
            *table(
                ("model", "AUPRC 95% CI", "AUROC 95% CI", "lift 95% CI"),
                [
                    [
                        name,
                        f"[{ci['auprc']['lo']:.4f}, {ci['auprc']['hi']:.4f}]",
                        f"[{ci['auroc']['lo']:.4f}, {ci['auroc']['hi']:.4f}]",
                        f"[{ci['lift']['lo']:.2f}, {ci['lift']['hi']:.2f}]",
                    ]
                    for name, ci in intervals.items()
                ],
            ),
            "",
            "Percentile intervals over "
            f"{next(iter(intervals.values()))['auprc']['resamples']:,} resamples of the holdout "
            "rows, against the point estimates in the table above.",
            "",
        ]

    ablation = evaluation.get("block_ablation") or {}
    if ablation:
        ordered = sorted(ablation.items(), key=lambda kv: kv[1]["holdout"]["auprc"])
        lines += [
            "## Leave one feature block out",
            "",
            *table(
                ("block removed", "columns", "holdout AUPRC", "delta"),
                [
                    [
                        name,
                        row["dropped"],
                        f"{row['holdout']['auprc']:.4f}",
                        f"{row['delta']['holdout_auprc']:+.4f}",
                    ]
                    for name, row in ordered
                ],
            ),
            "",
        ]

    if evaluation["sensitivity"]:
        lines += [
            "## Sensitivity",
            "",
            *table(
                (
                    "variant",
                    "rows",
                    "positives",
                    "base rate",
                    "AUPRC",
                    "delta",
                    "AUROC",
                    "delta",
                ),
                _sensitivity_rows(evaluation["sensitivity"]),
            ),
            "",
            "Holdout figures. Redrawing the negatives moves the base rate, and AUPRC is defined "
            "against it, so the AUPRC column is not comparable across the ratio variants. AUROC "
            f"is, and it moves by at most {_widest(evaluation['sensitivity'], 'auroc'):.4f} "
            "across them all.",
            "",
            *[
                f"- `{name}`: {block['question']}"
                for name, block in evaluation["sensitivity"].items()
            ],
            "",
        ]

    if evaluation.get("event_verification"):
        lines += _event_section(evaluation["event_verification"])

    lines += [
        "## Figures",
        "",
        *[f"![{path.stem}]({_relative(path, root)})" for path in figure_paths],
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
