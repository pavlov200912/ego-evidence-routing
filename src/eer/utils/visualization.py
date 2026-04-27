"""Utilities for exporting selected-frame visualizations."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from eer.data.frames import Frame


def _sanitize_slug(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:120] if cleaned else "item"


def save_frame_collage(
    frames: list[Frame],
    output_path: str | Path,
    *,
    cell_size: tuple[int, int] = (320, 180),
    padding: int = 8,
    max_cols: int = 4,
) -> Path | None:
    """Save a collage image of selected frames.

    Args:
        frames: Frames to include in the collage.
        output_path: Destination image path.
        cell_size: Width/height for each frame tile.
        padding: Pixels between tiles and around the border.
        max_cols: Maximum number of columns in the grid.

    Returns:
        Saved path, or ``None`` when frames is empty.
    """
    if not frames:
        return None

    cols = min(max_cols, len(frames))
    rows = int(math.ceil(len(frames) / cols))
    cell_w, cell_h = cell_size

    width = cols * cell_w + (cols + 1) * padding
    height = rows * cell_h + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), color=(24, 24, 24))

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        tile = ImageOps.fit(frame.image.convert("RGB"), (cell_w, cell_h), method=Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def save_selected_frame_artifacts(
    *,
    frames: list[Frame],
    output_dir: str | Path,
    question_id: str,
    tool_name: str,
    question: str,
    video_id: str,
) -> tuple[Path | None, Path | None]:
    """Save collage + metadata json for a selected frame set."""
    if not frames:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_slug = _sanitize_slug(question_id)
    t_slug = _sanitize_slug(tool_name)
    stem = f"{q_slug}__{t_slug}"

    collage_path = save_frame_collage(frames, output_dir / f"{stem}.jpg")

    meta_path = output_dir / f"{stem}.json"
    payload = {
        "question_id": question_id,
        "video_id": video_id,
        "tool": tool_name,
        "question": question,
        "n_selected": len(frames),
        "selected_frames": [
            {"index": f.index, "timestamp_s": f.timestamp_s} for f in frames
        ],
        "collage_path": str(collage_path) if collage_path is not None else None,
    }
    meta_path.write_text(json.dumps(payload, indent=2))

    return collage_path, meta_path