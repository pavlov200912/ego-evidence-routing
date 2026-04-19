"""Run ablation: each question × each tool, record results.

Usage:
    uv run python scripts/run_ablation.py --config configs/default.yaml \\
        --tools uniform clip motion --limit 20
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from eer.data.frames import extract_candidate_frames
from eer.data.hdepic import HDEPICDataset
from eer.tools.base import EvidenceTool
from eer.tools.clip_retrieval import CLIPRetrievalTool
from eer.tools.motion import MotionTool
from eer.tools.uniform import UniformTool
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: dict[str, type[EvidenceTool]] = {
    "uniform": UniformTool,
    "clip": CLIPRetrievalTool,
    "motion": MotionTool,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study: one result per question × tool.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument(
        "--tools",
        nargs="+",
        default=list(_TOOL_REGISTRY.keys()),
        choices=list(_TOOL_REGISTRY.keys()),
        help="Which tools to run (default: all).",
    )
    p.add_argument("--limit", type=int, default=None, help="Run on first N questions only.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def build_tools(tool_names: list[str], cfg: dict) -> list[EvidenceTool]:
    """Instantiate the requested tools, passing config where needed."""
    tools: list[EvidenceTool] = []
    for name in tool_names:
        cls = _TOOL_REGISTRY[name]
        if name == "clip":
            tools.append(
                CLIPRetrievalTool(
                    model_name=cfg["tools"]["clip_model"],
                    pretrained=cfg["tools"]["clip_pretrained"],
                )
            )
        else:
            tools.append(cls())
    return tools


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    cfg = yaml.safe_load(args.config.read_text())

    questions_path = Path(cfg["data"]["vqa_questions_path"])
    if questions_path.suffix == ".csv":
        dataset = HDEPICDataset.from_csv(questions_path)
    else:
        dataset = HDEPICDataset.from_json(questions_path)

    dataset = dataset.filter_by_duration(cfg["data"]["max_clip_duration_s"])
    questions = list(dataset)
    if args.limit is not None:
        questions = questions[: args.limit]

    logger.info("Running ablation on %d questions × %s tools", len(questions), args.tools)

    vlm = QwenVLM(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"],
    )
    tools = build_tools(args.tools, cfg)

    video_clips_dir = Path(cfg["data"]["video_clips_dir"])
    candidate_fps = cfg["data"]["candidate_fps"]
    frame_budget = cfg["tools"]["frame_budget"]

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"ablation_{timestamp}.csv"

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
            candidate_frames = extract_candidate_frames(
                video_path=video_path,
                fps=float(candidate_fps),
            )

            for tool in tools:
                selected = tool.select(
                    candidate_frames=candidate_frames,
                    question=q.question,
                    budget=frame_budget,
                )
                aux_images = [f.image for f in selected]

                result = vlm.answer_vqa(
                    video_path=video_path if video_path.exists() else None,
                    auxiliary_frames=aux_images if aux_images else None,
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
                        "tool": tool.name,
                        "predicted": result.predicted_letter,
                        "correct": q.correct_answer,
                        "log_prob": log_prob,
                        "is_correct": int(is_correct),
                    }
                )
                logger.info(
                    "Q %s | tool=%s: predicted=%s correct=%s (%s)",
                    q.question_id,
                    tool.name,
                    result.predicted_letter,
                    q.correct_answer,
                    "✓" if is_correct else "✗",
                )

    logger.info("Ablation results saved to %s", output_path)


if __name__ == "__main__":
    main()
