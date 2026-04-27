"""Run VQA on EgoTextVQA with optional evidence tools.

Usage:
    python scripts/run_egotextvqa_baseline.py --config configs/egotextvqa.yaml --limit 10 --tool ocr
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml

from eer.data.egotextvqa import EgoTextVQADataset
from eer.data.frames import extract_candidate_frames
from eer.utils.logging import setup_logging
from eer.vlm.qwen import QwenVLM

# Available Tools
from eer.tools.clip_retrieval import CLIPRetrievalTool
from eer.tools.motion import MotionTool
from eer.tools.ocr import OCRTool
from eer.tools.uniform import UniformTool
from eer.utils.visualization import save_selected_frame_artifacts

logger = logging.getLogger(__name__)


def _build_video_index(root_dir: Path) -> dict[str, Path]:
    """Build a stem -> path lookup for all .mp4 files under *root_dir*."""
    index: dict[str, Path] = {}
    duplicate_stems: defaultdict[str, int] = defaultdict(int)

    if not root_dir.exists() or not root_dir.is_dir():
        logger.warning("Video root directory does not exist or is not a directory: %s", root_dir)
        return index

    for video_path in root_dir.rglob("*.mp4"):
        stem = video_path.stem
        if stem in index:
            duplicate_stems[stem] += 1
            continue
        index[stem] = video_path

    if duplicate_stems:
        logger.warning(
            "Found %d duplicate video stems while indexing; using first match for each stem.",
            len(duplicate_stems),
        )

    logger.info("Indexed %d videos under %s", len(index), root_dir)
    return index


def _resolve_video_path(video_id: str, root_dir: Path, video_index: dict[str, Path]) -> Path | None:
    """Resolve a video path for *video_id* using direct and recursive lookup."""
    direct_mp4 = root_dir / f"{video_id}.mp4"
    if direct_mp4.exists():
        return direct_mp4

    direct_raw = root_dir / video_id
    if direct_raw.exists() and direct_raw.is_file():
        return direct_raw

    return video_index.get(video_id)

def load_tool(tool_name: str | None):
    if not tool_name or tool_name == "none" or tool_name == "baseline":
        return None
    if tool_name == "clip":
        return CLIPRetrievalTool()
    elif tool_name == "motion":
        return MotionTool()
    elif tool_name == "ocr":
        return OCRTool()
    elif tool_name == "uniform":
        return UniformTool()
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQA execution for EgoTextVQA.")
    p.add_argument("--config", type=Path, default=Path("configs/egotextvqa.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tool", type=str, default="baseline", choices=["baseline", "uniform", "clip", "motion", "ocr", "none"])
    p.add_argument("--frames-only", action="store_true", help="If set, only pass the tool-extracted frames to the VLM (omit the full video).")
    p.add_argument("--budget", type=int, default=8, help="Number of frames the tool can select.")
    p.add_argument("--fps", type=float, default=1.0, help="FPS extraction rate for candidate frames.")
    p.add_argument("--vlm-fps", type=float, default=1.0, help="FPS extraction rate for Qwen when reading the raw video.")
    p.add_argument(
        "--save-collages",
        action="store_true",
        help="Save collage images and metadata for selected evidence frames.",
    )
    p.add_argument(
        "--collages-dir",
        type=Path,
        default=None,
        help="Output directory for collages (default: <results_dir>/<run_name>_collages).",
    )
    p.add_argument(
        "--collage-max",
        type=int,
        default=None,
        help="Maximum number of collages to save.",
    )
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

    logger.info("Running %s on %d questions", args.tool, len(questions))

    # Load Tool
    evidence_tool = load_tool(args.tool)

    # Load VLM
    vlm = QwenVLM(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"],
    )

    video_clips_dir = Path(cfg["data"]["video_clips_dir"])
    video_index = _build_video_index(video_clips_dir)
    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{args.tool}_{timestamp}.csv"
    collage_root = (
        args.collages_dir
        if args.collages_dir is not None
        else results_dir / f"{args.tool}_{timestamp}_collages"
    )
    saved_collages = 0

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
            video_path = _resolve_video_path(str(q.video_id), video_clips_dir, video_index)
            if video_path is None:
                logger.warning("Video not found for video_id=%s under %s", q.video_id, video_clips_dir)
            
            auxiliary_images = None
            if evidence_tool is not None and video_path is not None:
                # Extract candidate frames (using `--fps` natively handles the heavy lifting)
                candidate_frames = extract_candidate_frames(video_path, fps=args.fps)
                
                # Use the tool to find the most relevant ones. Default budget is 8.
                selected_frames = evidence_tool.select(candidate_frames, q.question, budget=args.budget)

                if args.save_collages:
                    if args.collage_max is None or saved_collages < args.collage_max:
                        collage_path, _ = save_selected_frame_artifacts(
                            frames=selected_frames,
                            output_dir=collage_root,
                            question_id=q.question_id,
                            tool_name=args.tool,
                            question=q.question,
                            video_id=q.video_id,
                        )
                        if collage_path is not None:
                            saved_collages += 1
                
                # Extract PIL images to pass to the Vision-Language Model
                if selected_frames:
                    auxiliary_images = [f.image for f in selected_frames]
                    logger.info(f"Tool {args.tool} extracted {len(auxiliary_images)} frames as evidence.")

            try:
                # If frames-only is requested and we successfully extracted evidence, omit the raw video
                if args.frames_only and auxiliary_images is not None:
                    final_video_path = None
                else:
                    final_video_path = video_path

                predicted_text = vlm.answer_open_ended(
                    video_path=final_video_path,
                    auxiliary_frames=auxiliary_images,
                    question=q.question,
                    video_fps=args.vlm_fps,
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
                    "tool": args.tool,
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

    logger.info("%s results saved to %s", args.tool.capitalize(), output_path)
    if args.save_collages:
        logger.info("Saved %d collage artifacts to %s", saved_collages, collage_root)

if __name__ == "__main__":
    main()
