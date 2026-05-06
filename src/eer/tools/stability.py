"""Stability-based frame selection tool."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


def _motion_scores(frames: list[Frame]) -> np.ndarray:
    arrays = [
        np.array(f.image.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
        for f in frames
    ]
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


def _center_of_stable_run(scores: np.ndarray) -> int:
    """Return the index of the center frame of the longest run of scores at or
    below the median. A run of many consecutive low-motion frames is more
    reliably stable than a single isolated low-motion frame. Falls back to
    argmin when no run exists."""
    threshold = float(np.mean(scores))
    best_len, best_center = 0, None
    run_start = None
    for i, s in enumerate(scores.tolist()):
        if s < threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                length = i - run_start
                if length > best_len:
                    best_len = length
                    best_center = run_start + length // 2
                run_start = None
    if run_start is not None:
        length = len(scores) - run_start
        if length > best_len:
            best_center = run_start + length // 2
    return best_center if best_center is not None else int(np.argmin(scores))


def _temporally_diverse_stable(
    frames: list[Frame], scores: np.ndarray, budget: int
) -> list[Frame]:
    """Select the center of the longest stable run per temporal bucket."""
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
        idx = _center_of_stable_run(scores_t[start:end])
        selected.append(frames_t[start + idx])
    return selected


class StabilityTool(EvidenceTool):
    """Select the most temporally stable frames from a video clip, spread across time.

    Picks the lowest-motion frame within each temporal bucket. In egocentric
    video, low-motion frames correspond to moments where the camera has
    stabilised — typically when the wearer is looking at something rather than
    walking past it.
    """

    @property
    def name(self) -> str:
        return "stability"

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
        selected = _temporally_diverse_stable(candidate_frames, scores, budget)
        logger.debug("StabilityTool: selected %d/%d frames", len(selected), len(candidate_frames))
        return selected
