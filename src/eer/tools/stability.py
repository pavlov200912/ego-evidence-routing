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


def _temporally_diverse_stable(
    frames: list[Frame], scores: np.ndarray, budget: int
) -> list[Frame]:
    """Select lowest-motion frame per temporal bucket to ensure temporal spread."""
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
        best = int(np.argmin(scores_t[start:end]))
        selected.append(frames_t[start + best])
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
    ) -> list[Frame]:
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return sorted(candidate_frames, key=lambda f: f.timestamp_s)

        scores = _motion_scores(candidate_frames)
        selected = _temporally_diverse_stable(candidate_frames, scores, budget)
        logger.debug("StabilityTool: selected %d/%d frames", len(selected), len(candidate_frames))
        return selected
