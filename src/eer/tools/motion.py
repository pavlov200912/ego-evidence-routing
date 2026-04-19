"""Motion / scene-change keyframe selection tool."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


def _frame_to_gray_array(img: Image.Image, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Downsample and convert to float grayscale for fast diff computation."""
    return np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def _motion_scores(frames: list[Frame]) -> np.ndarray:
    """Compute a motion score for each frame using pixel-wise L1 differences.

    Each frame is scored as max(diff_with_prev, diff_with_next).
    Edge frames use only the available neighbor.

    Returns:
        Array of shape (N,) with non-negative motion scores.
    """
    arrays = [_frame_to_gray_array(f.image) for f in frames]
    n = len(arrays)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        diffs: list[float] = []
        if i > 0:
            diffs.append(float(np.mean(np.abs(arrays[i] - arrays[i - 1]))))
        if i < n - 1:
            diffs.append(float(np.mean(np.abs(arrays[i] - arrays[i + 1]))))
        scores[i] = max(diffs) if diffs else 0.0

    return scores


class MotionTool(EvidenceTool):
    """Select frames at high-motion / scene-change moments.

    Ranks frames by their pixel-wise L1 difference to their temporal
    neighbors and returns the top-K highest-motion frames in temporal order.
    """

    @property
    def name(self) -> str:
        return "motion"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Return top-*budget* high-motion frames from *candidate_frames*.

        Args:
            candidate_frames: All available frames for the clip.
            question: Not used by this tool.
            budget: Number of frames to return.

        Returns:
            Selected frames sorted by ascending timestamp.
        """
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return list(candidate_frames)

        scores = _motion_scores(candidate_frames)
        top_indices = np.argsort(scores)[::-1][:budget]

        selected = sorted(
            [candidate_frames[i] for i in top_indices],
            key=lambda f: f.timestamp_s,
        )
        logger.debug(
            "MotionTool: selected %d/%d frames", len(selected), len(candidate_frames)
        )
        return selected
