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
from pathlib import Path
from typing import Callable

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
_TOOL_NAMES = [
    "qwen_native",
    "uniform",
    "clip",
    "motion_then_clip",
    "ocr_crop",
    # "hand",
    "object_tracking",
    "uniform+clip",
    "answer_guided_oracle",
]
_DEFAULT_TOOL_NAMES = [
    "qwen_native",
    "uniform",
    "clip",
    "motion_then_clip",
    "ocr_crop",
    # "hand",
    "object_tracking",
    "uniform+clip",
    "answer_guided_oracle",
]
_TOOL_ALIASES = {}


def _cap_evidence(frames: list, budget: int) -> list:
    """Deduplicate and cap selected evidence to the shared visual budget."""
    if budget <= 0:
        return []

    seen: set[tuple[int, float]] = set()
    selected = []
    for frame in sorted(frames, key=lambda f: (f.timestamp_s, f.index)):
        key = (frame.index, frame.timestamp_s)
        if key in seen:
            continue
        seen.add(key)
        selected.append(frame)
        if len(selected) >= budget:
            break
    return selected

# Each entry is either:
#   tool_instance          → run with the shared --budget
#   _QWEN_NATIVE           → pass raw video only (C0)
def _build_tools(tool_names: list[str], cfg: dict) -> dict[str, object]:
    """Instantiate only the requested tools.

    Several tools load large models at construction time. Keeping this lazy
    makes small sanity runs cheap, e.g. ``--tools qwen_native,uniform_8`` will
    not load CLIP, EasyOCR, hands23, or GroundingDINO.
    """
    cache: dict[str, object] = {}

    def clip() -> CLIPRetrievalTool:
        if "clip" not in cache:
            cache["clip"] = CLIPRetrievalTool(
                model_name=cfg["tools"]["clip_model"],
                pretrained=cfg["tools"]["clip_pretrained"],
            )
        return cache["clip"]  # type: ignore[return-value]

    builders: dict[str, Callable[[], object]] = {
        "qwen_native": lambda: _QWEN_NATIVE,                                         # C0
        "uniform": lambda: UniformTool(),                                            # C1
        "clip": clip,                                                               # C3
        "motion_then_clip": lambda: CascadeTool(MotionTool(), clip(), overselect_factor=4),  # C4
        "ocr_crop": lambda: OCRCropTool(),                                           # C5
        "hand": lambda: HandTool(),                                                  # C6
        "object_tracking": lambda: ObjectTrackingTool(),                             # C7
        "uniform+clip": lambda: UnionTool(UniformTool(), clip()),                    # C8
        "answer_guided_oracle": lambda: AnswerGuidedTool(),
    }

    return {name: builders[name]() for name in tool_names}


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
    p.add_argument(
        "--tools", type=str, default=None,
        help="Comma-separated list of tools to run (e.g. 'qwen_native,object_tracking'). "
             "Defaults to the main non-oracle tools. Use answer_guided_oracle explicitly "
             "for the answer-choice time-window oracle.",
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

    cat_tag = (args.category or "all").replace("/", "_")
    results_dir = Path(cfg["eval"]["results_dir"]) / cat_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{mode}_k{args.budget}.csv"

    fieldnames = [
        "question_id",
        "category",
        "mode",
        "tool",
        "requested_budget",
        "n_selected_raw",
        "n_selected_final",
        "predicted",
        "correct",
        "is_correct",
    ]

    if args.tools is not None:
        requested = [_TOOL_ALIASES.get(t.strip(), t.strip()) for t in args.tools.split(",")]
        unknown = [t for t in requested if t not in _TOOL_NAMES]
        if unknown:
            logger.error(
                "Unknown tool(s): %s. Available: %s. Aliases: %s",
                unknown,
                _TOOL_NAMES,
                _TOOL_ALIASES,
            )
            return
    else:
        requested = list(_DEFAULT_TOOL_NAMES)

    if mode == "replace" and "qwen_native" in requested:
        requested = [name for name in requested if name != "qwen_native"]
        logger.info("Skipping qwen_native in replacement mode; it is identical to augmentation.")

    # Instantiate tools once, after filtering. CLIP is cached/shared across
    # conditions that need it.
    tool_instances = _build_tools(requested, cfg)

    correct_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

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
                tool = tool_entry
                tool_budget = args.budget
                n_selected_raw = 0
                n_selected_final = 0

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
                            n_selected_raw = len(selected)
                            selected = _cap_evidence(selected, tool_budget)
                            n_selected_final = len(selected)
                            if selected:
                                auxiliary_frames = selected
                            if n_selected_raw > n_selected_final:
                                logger.info(
                                    "Capped %s evidence from %d to %d items",
                                    tool_name,
                                    n_selected_raw,
                                    n_selected_final,
                                )
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
                    "requested_budget": tool_budget if tool is not _QWEN_NATIVE else 0,
                    "n_selected_raw": n_selected_raw,
                    "n_selected_final": n_selected_final,
                    "predicted": predicted,
                    "correct": q.correct_answer,
                    "is_correct": int(is_correct),
                })
                f.flush()
                logger.info(
                    "  [%-22s] budget=%d aux=%d/%d → %s  (correct=%s) %s",
                    tool_name,
                    tool_budget if tool is not _QWEN_NATIVE else 0,
                    n_selected_final,
                    n_selected_raw,
                    predicted,
                    q.correct_answer,
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
