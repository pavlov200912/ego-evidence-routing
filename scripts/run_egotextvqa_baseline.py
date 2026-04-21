"""Run the video-only baseline on EgoTextVQA.

Usage:
    uv run python scripts/run_egotextvqa_baseline.py --config configs/egotextvqa.yaml --limit 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import yaml

from eer.data.egotextvqa import EgoTextVQADataset
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-only baseline for EgoTextVQA.")
    p.add_argument("--config", type=Path, default=Path("configs/egotextvqa.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    cfg = yaml.safe_load(args.config.read_text())

    # Load dataset
    questions_path = Path(cfg["data"]["vqa_questions_path"])
    dataset = EgoTextVQADataset.from_jsonl(questions_path)

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
        "question_type",
        "tool",
        "predicted",
        "correct",
        "is_correct",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            video_path = video_clips_dir / f"{q.video_id}.mp4"
            
            try:
                predicted_text = vlm.answer_open_ended(
                    video_path=video_path if video_path.exists() else None,
                    auxiliary_frames=None,
                    question=q.question,
                )
            except Exception as e:
                logger.error(f"Error processing {q.question_id}: {e}")
                predicted_text = ""

            # VERY basic matching for open-ended QA.
            predicted_lower = predicted_text.lower()
            correct_lower = q.correct_answer.lower()
            is_correct = correct_lower in predicted_lower or predicted_lower in correct_lower

            writer.writerow(
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "tool": "baseline",
                    "predicted": predicted_text,
                    "correct": q.correct_answer,
                    "is_correct": int(is_correct),
                }
            )
            logger.info(
                "Q %s: predicted=%r correct=%r (%s)",
                q.question_id,
                predicted_text,
                q.correct_answer,
                "✓" if is_correct else "✗",
            )

    logger.info("Baseline results saved to %s", output_path)

if __name__ == "__main__":
    main()
