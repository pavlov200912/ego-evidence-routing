"""Produce summary tables and plots from ablation / routing CSVs.

Usage:
    uv run python scripts/analyze_results.py \\
        --ablation-csv results/ablation_20240101_120000.csv \\
        --output-dir results/analysis/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eer.eval.metrics import (
    compute_accuracy,
    compute_agreement,
    compute_oracle_routing,
    compute_per_category_accuracy,
    compute_per_prototype_accuracy,
)
from eer.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse ablation results and produce plots/tables.")
    p.add_argument(
        "--ablation-csv",
        type=Path,
        required=True,
        help="CSV produced by run_ablation.py.",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Optional baseline CSV to include in comparisons.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory for tables and plots.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved table: %s", path)


def _bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    results = pd.read_csv(args.ablation_csv)

    if args.baseline_csv is not None:
        baseline = pd.read_csv(args.baseline_csv)
        results = pd.concat([results, baseline], ignore_index=True)

    out = args.output_dir

    # Overall accuracy
    acc = compute_accuracy(results)
    acc_df = pd.DataFrame({"tool": list(acc.keys()), "accuracy": list(acc.values())})
    _save_table(acc_df, out / "overall_accuracy.csv")
    _bar_chart(acc_df, x="tool", y="accuracy", hue=None, title="Overall Accuracy by Tool", path=out / "overall_accuracy.png")

    # Per-category
    cat_df = compute_per_category_accuracy(results)
    _save_table(cat_df, out / "per_category_accuracy.csv")
    _bar_chart(cat_df, x="category", y="accuracy", hue="tool", title="Accuracy by Category", path=out / "per_category_accuracy.png")

    # Per-prototype
    proto_df = compute_per_prototype_accuracy(results)
    _save_table(proto_df, out / "per_prototype_accuracy.csv")

    # Oracle
    oracle = compute_oracle_routing(results)
    oracle_df = pd.DataFrame([oracle])
    _save_table(oracle_df, out / "oracle_routing.csv")
    logger.info("Oracle accuracy: %.3f  |  Baseline: %.3f", oracle["oracle"], oracle["baseline"])

    # Agreement analysis
    agree_df = compute_agreement(results)
    _save_table(agree_df, out / "agreement_accuracy.csv")
    _bar_chart(agree_df, x="n_tools_agree", y="accuracy", hue=None, title="Accuracy by Tool Agreement Level", path=out / "agreement_accuracy.png")

    logger.info("Analysis complete. Outputs in %s", out)


if __name__ == "__main__":
    main()
