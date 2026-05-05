"""Produce summary tables and plots from HD-EPIC ablation CSVs.

Accepts one or two CSVs (replacement mode and/or augmentation mode) produced
by run_hdepic_ablation.py. If both are provided they are merged and all plots
compare the two modes side by side.

Usage:
    # Single mode
    uv run python scripts/analyze_results.py \\
        --replace-csv results/hdepic/ablation_replace_20240101_120000.csv \\
        --output-dir results/analysis/

    # Both modes (side-by-side comparison)
    uv run python scripts/analyze_results.py \\
        --replace-csv results/hdepic/ablation_replace_*.csv \\
        --augment-csv results/hdepic/ablation_augment_*.csv \\
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
)
from eer.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse HD-EPIC ablation results.")
    p.add_argument("--replace-csv", type=Path, default=None,
                   help="CSV from replacement-mode run (no native video).")
    p.add_argument("--augment-csv", type=Path, default=None,
                   help="CSV from augmentation-mode run (native video + aux frames).")
    p.add_argument("--output-dir", type=Path, default=Path("results/analysis"),
                   help="Directory for tables and plots.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mode" not in df.columns:
        # Infer mode from filename if column is missing
        mode = "augment" if "augment" in path.stem else "replace"
        df["mode"] = mode
        logger.warning("Column 'mode' missing in %s — inferred as '%s'", path.name, mode)
    return df


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
    figsize: tuple[int, int] = (12, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
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

    if args.replace_csv is None and args.augment_csv is None:
        raise SystemExit("Provide at least one of --replace-csv or --augment-csv.")

    frames = []
    if args.replace_csv is not None:
        frames.append(_load(args.replace_csv))
    if args.augment_csv is not None:
        frames.append(_load(args.augment_csv))
    results = pd.concat(frames, ignore_index=True)

    out = args.output_dir
    has_both_modes = results["mode"].nunique() > 1

    # Overall accuracy per (tool, mode)
    acc_df = compute_accuracy(results)
    _save_table(acc_df, out / "overall_accuracy.csv")
    _bar_chart(
        acc_df, x="tool", y="accuracy",
        hue="mode" if has_both_modes else None,
        title="Overall Accuracy by Tool",
        path=out / "overall_accuracy.png",
        figsize=(14, 5),
    )

    # Per-category accuracy
    cat_df = compute_per_category_accuracy(results)
    _save_table(cat_df, out / "per_category_accuracy.csv")
    for mode, grp in cat_df.groupby("mode"):
        _bar_chart(
            grp, x="category", y="accuracy", hue="tool",
            title=f"Accuracy by Category — {mode}",
            path=out / f"per_category_accuracy_{mode}.png",
            figsize=(14, 5),
        )

    # If both modes: augment vs replace delta per tool
    if has_both_modes:
        pivot = acc_df.pivot(index="tool", columns="mode", values="accuracy")
        if "augment" in pivot.columns and "replace" in pivot.columns:
            pivot["delta"] = pivot["augment"] - pivot["replace"]
            pivot = pivot.reset_index().sort_values("delta", ascending=False)
            _save_table(pivot, out / "augment_vs_replace_delta.csv")
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.barplot(data=pivot, x="tool", y="delta", ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Accuracy gain: augment − replace")
            ax.set_ylabel("Δ Accuracy")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            path = out / "augment_vs_replace_delta.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("Saved plot: %s", path)

    # Oracle routing
    oracle_df = compute_oracle_routing(results)
    _save_table(oracle_df, out / "oracle_routing.csv")
    for _, row in oracle_df.iterrows():
        logger.info(
            "[%s] Oracle=%.3f | Best single tool: %s (%.3f)",
            row["mode"], row["oracle"], row["best_single_tool"], row["best_single_accuracy"],
        )

    # Agreement analysis
    agree_df = compute_agreement(results)
    _save_table(agree_df, out / "agreement_accuracy.csv")
    _bar_chart(
        agree_df, x="n_tools_agree", y="accuracy",
        hue="mode" if has_both_modes else None,
        title="Accuracy by Tool Agreement Level",
        path=out / "agreement_accuracy.png",
    )

    logger.info("Analysis complete. Outputs in %s", out)


if __name__ == "__main__":
    main()
