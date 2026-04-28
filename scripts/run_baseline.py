"""Run the video-only baseline on HD-EPIC VQA.

Usage:
    uv run python scripts/run_baseline.py --config configs/default.yaml --limit 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import yaml

from eer.data.hdepic import HDEPICDataset
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-only baseline for HD-EPIC VQA.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--limit", type=int, default=None, help="Run on first N questions only.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    cfg = yaml.safe_load(args.config.read_text())

    # Load dataset
    questions_path = Path(cfg["data"]["vqa_questions_path"])
    if questions_path.suffix == ".csv":
        dataset = HDEPICDataset.from_csv(questions_path)
    else:
        dataset = HDEPICDataset.from_json(questions_path)

    dataset = dataset.filter_by_duration(cfg["data"]["max_clip_duration_s"])

    questions = list(dataset)
    if args.limit is not None:
        questions = questions[: args.limit]

    logger.info("Running baseline on %d questions", len(questions))

    # Load VLM
    vlm = QwenVLM(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"],
    )

    video_clips_dir = Path(cfg["data"]["video_clips_dir"])
    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"baseline_{timestamp}.csv"

    fieldnames = [
        "question_id",
        "category",
        "prototype",
        "tool",
        "predicted",
        "correct",
        "log_prob",
        "is_correct",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            video_path = video_clips_dir / f"{q.video_id}.mp4"
            result = vlm.answer_multiple_choice(
                video_path=video_path if video_path.exists() else None,
                auxiliary_frames=None,
                question=q.question,
                choices=q.choices,
            )

            log_prob = result.log_probs.get(result.predicted_letter, float("nan"))
            is_correct = result.predicted_letter == q.correct_answer

            writer.writerow(
                {
                    "question_id": q.question_id,
                    "category": q.category,
                    "prototype": q.prototype,
                    "tool": "baseline",
                    "predicted": result.predicted_letter,
                    "correct": q.correct_answer,
                    "log_prob": log_prob,
                    "is_correct": int(is_correct),
                }
            )
            logger.info(
                "Q %s: predicted=%s correct=%s (%s)",
                q.question_id,
                result.predicted_letter,
                q.correct_answer,
                "✓" if is_correct else "✗",
            )

    logger.info("Baseline results saved to %s", output_path)


if __name__ == "__main__":
    main()
