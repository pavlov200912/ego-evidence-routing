"""Run oracle and predicted routing experiments.

Usage:
    uv run python scripts/run_routing.py --config configs/default.yaml \\
        --ablation-csv results/ablation_20240101_120000.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from eer.eval.metrics import compute_accuracy, compute_oracle_routing
from eer.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle and predicted routing evaluation (stubs for Week 3)."
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument(
        "--ablation-csv",
        type=Path,
        required=True,
        help="CSV produced by run_ablation.py.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    cfg = yaml.safe_load(args.config.read_text())  # noqa: F841

    logger.info("Loading ablation results from %s", args.ablation_csv)
    results = pd.read_csv(args.ablation_csv)

    # --- Oracle upper bound (computable from ablation results) ---
    oracle = compute_oracle_routing(results)
    logger.info("Oracle accuracy:   %.3f", oracle["oracle"])
    logger.info("Baseline accuracy: %.3f", oracle["baseline"])

    per_tool = compute_accuracy(results)
    for tool, acc in sorted(per_tool.items()):
        logger.info("  %-20s %.3f", tool, acc)

    # --- Predicted router (Week 3 stub) ---
    logger.info(
        "PredictedRouter not yet implemented — skipping. "
        "Implement routing/predicted.py in Week 3."
    )


if __name__ == "__main__":
    main()
