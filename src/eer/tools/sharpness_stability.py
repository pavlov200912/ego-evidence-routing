"""Hybrid sharpness + stability frame selection tool."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


def _sharpness_scores(frames: list[Frame]) -> np.ndarray:
    scores = np.zeros(len(frames), dtype=np.float32)
    for i, f in enumerate(frames):
        gray = np.array(f.image.convert("L"), dtype=np.float32)
        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        scores[i] = laplacian.var()
    return scores


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


def _normalize(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def _temporally_diverse(
    frames: list[Frame], scores: np.ndarray, budget: int
) -> list[Frame]:
    """Select highest-scoring frame per temporal bucket to ensure temporal spread."""
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
        best = int(np.argmax(scores_t[start:end]))
        selected.append(frames_t[start + best])
    return selected


class SharpnessStabilityTool(EvidenceTool):
    """Select frames that are both sharp and temporally stable, spread across time.

    Combines sharpness (Laplacian variance) and stability (inverse motion)
    as a weighted sum of their normalized values. Using addition rather than
    multiplication avoids zeroing out frames that score well on one dimension
    but near-zero on the other.
    """

    @property
    def name(self) -> str:
        return "sharpness_stability"

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

        sharpness = _normalize(_sharpness_scores(candidate_frames))
        stability = 1.0 - _normalize(_motion_scores(candidate_frames))
        combined = 0.5 * sharpness + 0.5 * stability

        selected = _temporally_diverse(candidate_frames, combined, budget)
        logger.debug(
            "SharpnessStabilityTool: selected %d/%d frames", len(selected), len(candidate_frames)
        )
        return selected
