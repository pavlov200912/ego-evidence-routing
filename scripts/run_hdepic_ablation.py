"""Tool-comparison ablation on HD-EPIC VQA.

Runs all evidence tools (replacement mode by default, augmentation mode with
--augment) on a subset of HD-EPIC questions and records per-question results.

Usage:
    # Quick sanity check — 4 questions, replacement mode
    uv run python scripts/run_hdepic_ablation.py \\
        --config configs/hdepic.yaml \\
        --category recipe_step_localization \\
        --limit 4

    # Full ablation — all long-video questions, both modes
    uv run python scripts/run_hdepic_ablation.py --config configs/hdepic.yaml
    uv run python scripts/run_hdepic_ablation.py --config configs/hdepic.yaml --augment

Category names match the JSON filenames in vqa-benchmark/, e.g.:
    fine_grained_action_localization
    ingredient_ingredient_adding_localization
    recipe_step_localization
    object_motion_object_movement_itinerary
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from eer.data.frames import extract_candidate_frames
from eer.data.hdepic import HDEPICDataset
from eer.tools.answer_guided import AnswerGuidedTool
from eer.tools.cascade import CascadeTool
from eer.tools.clip_retrieval import CLIPRetrievalTool
from eer.tools.crop import OCRCropTool
from eer.tools.hand import HandTool
from eer.tools.motion import MotionTool
from eer.tools.object_tracking import ObjectTrackingTool
from eer.tools.uniform import UniformTool
from eer.tools.union import UnionTool
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

logger = logging.getLogger(__name__)

_QWEN_NATIVE = object()  # sentinel: pass raw video, no auxiliary frames

# Each entry is either:
#   tool_instance          → run with the default budget
#   (tool_instance, int)   → run with the given budget override (e.g. uniform-16)
#   _QWEN_NATIVE           → pass raw video only (C0)
def _make_tools() -> dict[str, object]:
    clip = CLIPRetrievalTool()
    return {
        "qwen_native":     _QWEN_NATIVE,                                          # C0
        "uniform_8":       UniformTool(),                                          # C1
        "uniform_16":      (UniformTool(), 16),                                    # C2
        "clip":            clip,                                                   # C3
        "motion":          CascadeTool(MotionTool(), clip, overselect_factor=4),   # C4
        "ocr_crop":        OCRCropTool(),                                          # C5
        "hand":            HandTool(),                                             # C6
        "object_tracking": ObjectTrackingTool(),                                   # C7
        "uniform+clip":    UnionTool(UniformTool(), clip),                         # C8
        "answer_guided":   AnswerGuidedTool(),
    }


def _build_video_index(root_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root_dir.exists():
        logger.warning("Video dir does not exist: %s", root_dir)
        return index
    for p in root_dir.rglob("*.mp4"):
        if p.stem not in index:
            index[p.stem] = p
    logger.info("Indexed %d videos under %s", len(index), root_dir)
    return index


def _resolve_video(video_id: str, root: Path, index: dict[str, Path]) -> Path | None:
    direct = root / f"{video_id}.mp4"
    if direct.exists():
        return direct
    return index.get(video_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/hdepic.yaml"))
    p.add_argument(
        "--category", type=str, default=None,
        help="Filter to one category (JSON filename stem). If omitted, loads all categories.",
    )
    p.add_argument("--limit", type=int, default=None, help="Run on first N questions only.")
    p.add_argument("--budget", type=int, default=8, help="Default frame budget per tool.")
    p.add_argument("--fps", type=float, default=1.0, help="Candidate frame extraction rate.")
    p.add_argument(
        "--max-candidates", type=int, default=500,
        help="Cap candidate frames per question (uniform subsample). Prevents RAM OOM.",
    )
    p.add_argument("--vlm-fps", type=float, default=1.0, help="FPS passed to Qwen for the raw video.")
    p.add_argument("--vlm-max-frames", type=int, default=64,
                   help="Max frames Qwen samples from the raw video.")
    p.add_argument(
        "--augment", action="store_true",
        help="Augmentation mode: pass native video + auxiliary frames to Qwen. "
             "Default (no flag) is replacement mode: auxiliary frames only.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None, help="Write logs to this file in addition to stderr.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    cfg = yaml.safe_load(args.config.read_text())
    vqa_dir = Path(cfg["data"]["vqa_benchmark_dir"])
    video_clips_dir = Path(cfg["data"]["video_clips_dir"])

    categories = [args.category] if args.category else None
    dataset = HDEPICDataset.from_dir(vqa_dir, categories=categories)
    if "min_clip_duration_s" in cfg["data"]:
        dataset = dataset.filter_by_min_duration(cfg["data"]["min_clip_duration_s"])
    dataset = dataset.filter_by_duration(cfg["data"]["max_clip_duration_s"])

    questions = list(dataset)
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        logger.error("No questions found. Check --category and data paths.")
        return

    mode = "augment" if args.augment else "replace"
    logger.info(
        "Ablation mode=%s | %d questions | category=%s",
        mode, len(questions), args.category or "all",
    )

    vlm = QwenVLM(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"],
    )

    video_index = _build_video_index(video_clips_dir)

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cat_tag = (args.category or "all").replace("/", "_")
    output_path = results_dir / f"ablation_{mode}_{cat_tag}_{timestamp}.csv"

    fieldnames = ["question_id", "category", "mode", "tool", "predicted", "correct", "is_correct"]

    # Instantiate tools once — CLIP and GroundingDINO load large models
    tool_instances: dict[str, object] = _make_tools()

    correct_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            logger.info("--- Q %s [%s] ---", q.question_id, q.category)
            logger.info("    %s", q.question)

            video_path = _resolve_video(q.video_id, video_clips_dir, video_index)
            if video_path is None:
                logger.warning("Video not found: %s — skipping", q.video_id)
                continue

            candidate_frames = extract_candidate_frames(
                video_path, fps=args.fps,
                start_s=q.start_time, end_s=q.end_time,
            )
            if args.max_candidates and len(candidate_frames) > args.max_candidates:
                step = len(candidate_frames) / args.max_candidates
                candidate_frames = [candidate_frames[int(i * step)] for i in range(args.max_candidates)]
                logger.info("Subsampled candidate frames to %d", len(candidate_frames))

            for tool_name, tool_entry in tool_instances.items():
                # Unpack optional budget override
                if isinstance(tool_entry, tuple):
                    tool, tool_budget = tool_entry
                else:
                    tool, tool_budget = tool_entry, args.budget

                if tool is _QWEN_NATIVE:
                    auxiliary_frames = None
                    final_video_path = video_path
                else:
                    auxiliary_frames = None
                    if candidate_frames:
                        try:
                            selected = tool.select(
                                candidate_frames,
                                q.question,
                                budget=tool_budget,
                                choices=q.choices,
                                video_path=str(video_path),
                            )
                            if selected:
                                auxiliary_frames = selected
                        except Exception as e:
                            logger.error(
                                "Frame selection error Q %s tool %s: %s",
                                q.question_id, tool_name, e,
                            )
                    # Augmentation mode: keep native video alongside auxiliary frames
                    final_video_path = video_path if args.augment else None

                try:
                    result = vlm.answer_multiple_choice(
                        video_path=final_video_path,
                        auxiliary_frames=auxiliary_frames,
                        question=q.question,
                        choices=q.choices,
                        video_fps=args.vlm_fps,
                        video_max_frames=args.vlm_max_frames,
                    )
                    predicted = result.predicted_letter
                except Exception as e:
                    logger.error("VLM error Q %s tool %s: %s", q.question_id, tool_name, e)
                    predicted = "?"
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                is_correct = predicted == q.correct_answer
                correct_counts[tool_name] += int(is_correct)
                total_counts[tool_name] += 1

                writer.writerow({
                    "question_id": q.question_id,
                    "category": q.category,
                    "mode": mode,
                    "tool": tool_name,
                    "predicted": predicted,
                    "correct": q.correct_answer,
                    "is_correct": int(is_correct),
                })
                logger.info(
                    "  [%-22s] → %s  (correct=%s) %s",
                    tool_name, predicted, q.correct_answer,
                    "✓" if is_correct else "✗",
                )

    # Summary table
    print(f"\nMode: {mode}")
    print(f"{'Tool':<22} {'Correct':>7} {'Total':>7} {'Accuracy':>10}")
    print("-" * 50)
    for tool_name in tool_instances:
        n = total_counts[tool_name]
        c = correct_counts[tool_name]
        acc = f"{100*c/n:.1f}%" if n > 0 else "—"
        print(f"{tool_name:<22} {c:>7} {n:>7} {acc:>10}")

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
