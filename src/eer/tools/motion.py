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


def _settled_after_peak(scores: np.ndarray, settle_window: int = 4) -> int:
    """Find the scene-change peak (argmax), then return the most stable frame
    in the window immediately following it.

    The peak frame itself is often mid-transition and blurry. The frame a few
    steps later, once motion has subsided, captures the new scene content more
    cleanly. Falls back to the peak when it sits at the end of the bucket."""
    peak = int(np.argmax(scores))
    search_start = peak + 1
    search_end = min(peak + 1 + settle_window, len(scores))
    if search_start >= len(scores):
        return peak
    return search_start + int(np.argmin(scores[search_start:search_end]))


def _temporally_diverse(
    frames: list[Frame], scores: np.ndarray, budget: int
) -> list[Frame]:
    """Select the settle point after the motion peak per temporal bucket."""
    pairs = sorted(zip(frames, scores.tolist()), key=lambda x: x[0].timestamp_s)
    frames_t = [p[0] for p in pairs]
    scores_t = np.array([p[1] for p in pairs])
    n = len(frames_t)
    bucket_size = n / budget
    selected = []
    for b in range(budget):
        start = int(b * bucket_size)
        end = min(int((b + 1) * bucket_size), n)
        if start >= end:
            continue
        idx = _settled_after_peak(scores_t[start:end])
        selected.append(frames_t[start + idx])
    return selected


class MotionTool(EvidenceTool):
    """Select frames at high-motion / scene-change moments, spread across time.

    Picks the highest inter-frame motion (L1 pixel diff) frame within each
    temporal bucket, ensuring coverage of scene changes across the full clip
    rather than clustering on a single high-activity segment.
    """

    @property
    def name(self) -> str:
        return "motion"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        **_kwargs,
    ) -> list[Frame]:
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return sorted(candidate_frames, key=lambda f: f.timestamp_s)

        scores = _motion_scores(candidate_frames)
        selected = _temporally_diverse(candidate_frames, scores, budget)
        logger.debug(
            "MotionTool: selected %d/%d frames", len(selected), len(candidate_frames)
        )
        return selected
