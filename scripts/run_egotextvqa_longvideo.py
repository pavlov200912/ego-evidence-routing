"""Long-video ablation for EgoTextVQA.

Builds synthetic long videos by concatenating a target clip with N-1 distractor
clips drawn randomly from the dataset. This forces tools to localise the relevant
moment rather than scanning a short pre-clipped window.

Usage:
    uv run python scripts/run_egotextvqa_longvideo.py \
        --config configs/egotextvqa.yaml \
        --n-clips 5 \
        --tools uniform sharpness motion sharpness_motion timestamp \
        --limit 20 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from datetime import datetime
from pathlib import Path

import yaml

from eer.data.egotextvqa import EgoTextVQADataset, EgoTextVQAQuestion
from eer.data.frames import Frame, extract_candidate_frames
from eer.tools.base import EvidenceTool
from eer.tools.clip_retrieval import CLIPRetrievalTool
from eer.tools.motion import MotionTool
from eer.tools.sharpness import SharpnessTool
from eer.tools.sharpness_motion import SharpnessMotionTool
from eer.tools.sharpness_stability import SharpnessStabilityTool
from eer.tools.stability import StabilityTool
from eer.tools.timestamp import TimestampTool
from eer.tools.uniform import UniformTool
from eer.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: dict[str, type[EvidenceTool]] = {
    "uniform": UniformTool,
    "motion": MotionTool,
    "sharpness": SharpnessTool,
    "sharpness_motion": SharpnessMotionTool,
    "stability": StabilityTool,
    "sharpness_stability": SharpnessStabilityTool,
    "timestamp": TimestampTool,
    "clip": CLIPRetrievalTool,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/egotextvqa.yaml"))
    p.add_argument("--n-clips", type=int, default=5,
                   help="Total number of clips in the synthetic long video (1 target + N-1 distractors).")
    p.add_argument("--tools", nargs="+", default=["uniform", "sharpness", "motion", "timestamp"],
                   choices=list(_TOOL_REGISTRY.keys()))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def build_tools(tool_names: list[str], cfg: dict) -> list[EvidenceTool]:
    tools: list[EvidenceTool] = []
    for name in tool_names:
        if name == "clip":
            tools.append(CLIPRetrievalTool(
                model_name=cfg["tools"]["clip_model"],
                pretrained=cfg["tools"]["clip_pretrained"],
            ))
        else:
            tools.append(_TOOL_REGISTRY[name]())
    return tools


def _clip_duration(frames: list[Frame]) -> float:
    """Estimate clip duration as last frame timestamp + 1 frame gap (at 1 fps)."""
    if not frames:
        return 10.0
    return frames[-1].timestamp_s + 1.0


def build_long_video(
    target_frames: list[Frame],
    distractor_frame_lists: list[list[Frame]],
    target_position: int,
) -> tuple[list[Frame], float]:
    """Concatenate clips into a single long frame list.

    Places *target_frames* at *target_position* among the distractors.

    Returns:
        (long_frames, target_timestamp_offset) — the offset to add to q.timestamp
        to get the correct timestamp in the long video.
    """
    ordered_clips = (
        distractor_frame_lists[:target_position]
        + [target_frames]
        + distractor_frame_lists[target_position:]
    )

    long_frames: list[Frame] = []
    target_offset = 0.0
    global_offset = 0.0
    global_idx = 0

    for clip_pos, clip_frames in enumerate(ordered_clips):
        if clip_pos == target_position:
            target_offset = global_offset
        for f in clip_frames:
            long_frames.append(Frame(
                index=global_idx,
                timestamp_s=f.timestamp_s + global_offset,
                image=f.image,
            ))
            global_idx += 1
        global_offset += _clip_duration(clip_frames)

    return long_frames, target_offset


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    random.seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text())

    dataset = EgoTextVQADataset.from_jsonl(Path(cfg["data"]["vqa_questions_path"]))
    questions = list(dataset)
    if args.limit is not None:
        questions = questions[: args.limit]

    video_clips_dir = Path(cfg["data"]["video_clips_dir"])
    frame_budget = cfg.get("tools", {}).get("frame_budget", 4)
    n_clips = args.n_clips
    n_distractors = n_clips - 1

    # Build a lookup: video_id → list of questions (to pick distractors from other videos)
    all_video_ids = list({q.video_id for q in dataset})

    logger.info(
        "Long-video ablation: %d questions × %d tools | n_clips=%d | budget=%d",
        len(questions), len(args.tools), n_clips, frame_budget,
    )

    vlm_cfg = cfg["model"]
    from eer.vlm.qwen import QwenVLM
    vlm = QwenVLM(
        model_name=vlm_cfg["name"],
        dtype=vlm_cfg["dtype"],
        device_map=vlm_cfg["device_map"],
    )
    tools = build_tools(args.tools, cfg)

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools_str = "_".join(args.tools)
    output_path = results_dir / f"longvideo_n{n_clips}_{tools_str}_{timestamp_str}.csv"

    fieldnames = [
        "question_id", "question_type", "tool", "n_clips",
        "target_position", "n_total_frames", "predicted", "correct", "is_correct",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            target_video_path = video_clips_dir / f"{q.video_id}.mp4"
            target_frames = (
                extract_candidate_frames(target_video_path, fps=1.0)
                if target_video_path.exists() else []
            )
            if not target_frames:
                logger.warning("Q %s: target video missing, skipping.", q.question_id)
                continue

            # Sample distractor video ids (different from target)
            distractor_ids = random.sample(
                [vid for vid in all_video_ids if vid != q.video_id],
                k=min(n_distractors, len(all_video_ids) - 1),
            )
            distractor_frame_lists: list[list[Frame]] = []
            for vid in distractor_ids:
                path = video_clips_dir / f"{vid}.mp4"
                frames = extract_candidate_frames(path, fps=1.0) if path.exists() else []
                distractor_frame_lists.append(frames)

            # Place target at a random position among distractors
            target_position = random.randint(0, len(distractor_frame_lists))

            long_frames, target_offset = build_long_video(
                target_frames, distractor_frame_lists, target_position
            )

            logger.info(
                "Q %s: long video = %d frames (~%.0fs) | target at position %d/%d (offset=%.1fs)",
                q.question_id, len(long_frames), long_frames[-1].timestamp_s,
                target_position, n_clips, target_offset,
            )

            selections: dict[str, set[int]] = {}

            for tool in tools:
                if hasattr(tool, "timestamp_s"):
                    tool.timestamp_s = q.timestamp + target_offset
                selected = tool.select(long_frames, question=q.question, budget=frame_budget)
                selections[tool.name] = {f.index for f in selected}

                logger.info(
                    "Q %s | tool=%s: selected %d/%d frames %s",
                    q.question_id, tool.name, len(selected), len(long_frames),
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
                    logger.error("Q %s | tool=%s: inference error: %s", q.question_id, tool.name, e)
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
                    "n_clips": n_clips,
                    "target_position": target_position,
                    "n_total_frames": len(long_frames),
                    "predicted": predicted_text,
                    "correct": q.correct_answer,
                    "is_correct": int(is_correct),
                })
                logger.info(
                    "Q %s | tool=%s: predicted=%r correct=%r (%s)",
                    q.question_id, tool.name, predicted_text, q.correct_answer,
                    "✓" if is_correct else "✗",
                )

            # Pairwise overlap between tool selections
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

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
