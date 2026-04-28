"""Run ablation study on EgoTextVQA: each question × each tool.

Usage:
    uv run python scripts/run_egotextvqa_ablation.py --config configs/egotextvqa.yaml --tools motion uniform --limit 20
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import yaml

from eer.data.egotextvqa import EgoTextVQADataset
from eer.data.frames import extract_candidate_frames
from eer.tools.base import EvidenceTool
from eer.tools.cascade import CascadeTool
from eer.tools.clip_retrieval import CLIPRetrievalTool
from eer.tools.motion import MotionTool
from eer.tools.ocr import OCRTool
from eer.tools.sharpness_stability import SharpnessStabilityTool
from eer.tools.stability import StabilityTool
from eer.tools.uniform import UniformTool
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: dict[str, type[EvidenceTool]] = {
    "uniform": UniformTool,
    "clip": CLIPRetrievalTool,
    "motion": MotionTool,
    "stability": StabilityTool,
    "sharpness_stability": SharpnessStabilityTool,
    "ocr": OCRTool,
    # cascade combinations (instantiated in build_tools)
    "clip_then_sharpness_stability": CascadeTool,
    "ocr_then_sharpness_stability": CascadeTool,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study on EgoTextVQA: one result per question × tool.")
    p.add_argument("--config", type=Path, default=Path("configs/egotextvqa.yaml"))
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
    tools: list[EvidenceTool] = []
    for name in tool_names:
        if name == "clip" or name.startswith("clip_then_"):
            clip = CLIPRetrievalTool(
                model_name=cfg["tools"]["clip_model"],
                pretrained=cfg["tools"]["clip_pretrained"],
            )
            if name == "clip":
                tools.append(clip)
            elif name == "clip_then_sharpness_stability":
                tools.append(CascadeTool(clip, SharpnessStabilityTool()))
        elif name == "ocr_then_sharpness_stability":
            tools.append(CascadeTool(OCRTool(), SharpnessStabilityTool()))
        else:
            tools.append(_TOOL_REGISTRY[name]())
    return tools


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    cfg = yaml.safe_load(args.config.read_text())

    dataset = EgoTextVQADataset.from_jsonl(Path(cfg["data"]["vqa_questions_path"]))
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
    frame_budget = cfg.get("tools", {}).get("frame_budget", 8)

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools_str = "_".join(args.tools)
    output_path = results_dir / f"ablation_{tools_str}_{timestamp}.csv"

    fieldnames = ["question_id", "question_type", "tool", "predicted", "correct", "is_correct"]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            video_path = video_clips_dir / f"{q.video_id}.mp4"
            candidate_frames = extract_candidate_frames(video_path, fps=1.0) if video_path.exists() else []
            n_candidates = len(candidate_frames)

            selections: dict[str, set[int]] = {}

            for tool in tools:
                if hasattr(tool, "timestamp_s"):
                    tool.timestamp_s = q.timestamp
                selected = tool.select(candidate_frames, question=q.question, budget=frame_budget)
                selections[tool.name] = {f.index for f in selected}

                logger.info(
                    "Q %s | tool=%s: selected %d/%d frames %s",
                    q.question_id,
                    tool.name,
                    len(selected),
                    n_candidates,
                    sorted(f.index for f in selected),
                )

                aux_images = [f.image for f in selected] or None

                try:
                    predicted_text = vlm.answer_open_ended(
                        video_path=None,
                        auxiliary_frames=aux_images,
                        question=q.question,
                    )
                except Exception as e:
                    logger.error("Error processing %s with tool %s: %s", q.question_id, tool.name, e)
                    predicted_text = ""

                predicted_lower = predicted_text.lower().strip()
                correct_lower = q.correct_answer.lower().strip()
                is_correct = bool(predicted_lower) and (
                    correct_lower in predicted_lower or predicted_lower in correct_lower
                )

                writer.writerow({
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "tool": tool.name,
                    "predicted": predicted_text,
                    "correct": q.correct_answer,
                    "is_correct": int(is_correct),
                })
                logger.info(
                    "Q %s | tool=%s: predicted=%r correct=%r (%s)",
                    q.question_id,
                    tool.name,
                    predicted_text,
                    q.correct_answer,
                    "✓" if is_correct else "✗",
                )

            # Log pairwise overlap between tool selections for this question
            tool_names = list(selections.keys())
            for i in range(len(tool_names)):
                for j in range(i + 1, len(tool_names)):
                    a, b = tool_names[i], tool_names[j]
                    inter = len(selections[a] & selections[b])
                    union = len(selections[a] | selections[b])
                    jaccard = inter / union if union else 1.0
                    logger.info(
                        "Q %s | overlap %s∩%s: %d/%d frames (Jaccard=%.2f)",
                        q.question_id, a, b, inter, frame_budget, jaccard,
                    )

    logger.info("Ablation results saved to %s", output_path)


if __name__ == "__main__":
    main()
