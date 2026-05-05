"""Isolated test for ObjectTrackingTool on a real HD-EPIC question.

Does NOT load Qwen — only tests GroundingDINO frame selection.
Saves selected frames as JPEG for visual inspection.

Usage:
    uv run python scripts/test_object_tracking.py
    uv run python scripts/test_object_tracking.py --category ingredient_ingredient_adding_localization
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from eer.data.frames import extract_candidate_frames
from eer.data.hdepic import HDEPICDataset
from eer.tools.object_tracking import ObjectTrackingTool, _extract_object_query
from eer.utils.logging import setup_logging

os.environ.setdefault("HF_HOME", "/scratch/izar/cljordan/cache/huggingface")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/hdepic.yaml"))
    p.add_argument("--category", type=str, default="fine_grained_action_localization")
    p.add_argument("--question-idx", type=int, default=0, help="Index of question to test")
    p.add_argument("--max-candidates", type=int, default=150, help="Cap frames for speed")
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("results/ot_test"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.DEBUG)

    cfg = yaml.safe_load(args.config.read_text())
    vqa_dir = Path(cfg["data"]["vqa_benchmark_dir"])
    video_dir = Path(cfg["data"]["video_clips_dir"])

    dataset = HDEPICDataset.from_dir(vqa_dir, categories=[args.category])
    questions = list(dataset)
    if not questions:
        logger.error("No questions found for category %s", args.category)
        return

    q = questions[args.question_idx]
    logger.info("Question: %s", q.question)
    logger.info("Choices: %s", q.choices)
    logger.info("Correct: %s", q.correct_answer)
    logger.info("Video: %s  window: %s – %s", q.video_id, q.start_time, q.end_time)

    query = _extract_object_query(q.question)
    logger.info("Extracted query: %r", query)

    # Locate video
    video_path = None
    for p in video_dir.rglob("*.mp4"):
        if q.video_id in p.stem or q.video_id == p.stem:
            video_path = p
            break
    if video_path is None:
        # Try subdirectory match
        parts = q.video_id.split("-")
        if parts:
            for p in (video_dir / parts[0]).glob("*.mp4"):
                if q.video_id in p.stem:
                    video_path = p
                    break
    if video_path is None:
        logger.error("Video not found for %s", q.video_id)
        return
    logger.info("Video path: %s", video_path)

    # Extract candidate frames
    candidates = extract_candidate_frames(
        video_path, fps=cfg["data"]["candidate_fps"],
        start_s=q.start_time, end_s=q.end_time,
    )
    logger.info("Extracted %d candidate frames", len(candidates))

    if len(candidates) > args.max_candidates:
        step = len(candidates) / args.max_candidates
        candidates = [candidates[int(i * step)] for i in range(args.max_candidates)]
        logger.info("Subsampled to %d frames", len(candidates))

    # Run ObjectTrackingTool — with score logging
    tool = ObjectTrackingTool()

    # Score all frames manually so we can log them
    if query is not None:
        scores = tool._score_frames(candidates, query)
        n_detected = int((scores > 0).sum())
        logger.info(
            "Detections: %d/%d frames scored > 0  |  max=%.3f  mean_nonzero=%.3f",
            n_detected, len(candidates),
            float(scores.max()),
            float(scores[scores > 0].mean()) if n_detected > 0 else 0.0,
        )
        # Log top-10 by score
        top10 = np.argsort(scores)[::-1][:10]
        logger.info("Top-10 frames by detection score:")
        for rank, idx in enumerate(top10):
            logger.info("  #%d  t=%.1fs  score=%.3f", rank + 1, candidates[idx].timestamp_s, scores[idx])
    else:
        logger.info("No text query — uniform fallback will be used")

    selected = tool.select(candidates, q.question, budget=args.budget)
    logger.info(
        "Selected %d frames: %s",
        len(selected), [f"{f.timestamp_s:.1f}s" for f in selected],
    )

    # Save selected frames for visual inspection
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(selected):
        out = args.out_dir / f"frame_{i:02d}_t{frame.timestamp_s:.1f}s.jpg"
        frame.image.save(out)
    logger.info("Saved %d frames to %s", len(selected), args.out_dir)


if __name__ == "__main__":
    main()
