"""Object tracking evidence tool using GroundingDINO (via transformers)."""

from __future__ import annotations

import logging
import re

import numpy as np
import torch

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)

_BBOX_RE = re.compile(r"<BBOX[^>]*>")
_TIME_RE = re.compile(r"<TIME[^>]*>")
_ACTION_BRACKET_RE = re.compile(r"<(?!TIME\b)(?!BBOX\b)([^>]+)>")
_INGREDIENT_RE = re.compile(
    r"\bingredient\s+([\w][\w\s,]+?)(?:\s+added|\s+used|\s+weigh|\s+to\b)",
    re.IGNORECASE,
)


def _extract_object_query(question: str) -> str | None:
    """Extract a grounding text query from the question.

    Returns None if no useful text query can be formed (e.g. BBOX-only
    object_motion questions where the object has no textual name).
    """
    # BBOX-only questions — object identified visually, not by name
    if _BBOX_RE.search(question) and not _INGREDIENT_RE.search(question):
        return None

    # "ingredient salt added to recipe …" → "salt"
    m = _INGREDIENT_RE.search(question)
    if m:
        return m.group(1).strip().rstrip(",")

    # "action <hit spatula>" → last token is usually the object
    m = _ACTION_BRACKET_RE.search(question)
    if m:
        words = m.group(1).split()
        return words[-1] if words else None

    # Generic: strip TIME/BBOX tokens and use the cleaned question text
    cleaned = _TIME_RE.sub("", _BBOX_RE.sub("", question)).strip()
    return cleaned[:200] if cleaned else None


class ObjectTrackingTool(EvidenceTool):
    """Select frames where the queried object is most visible using GroundingDINO.

    Uses the transformers implementation of grounding-dino-tiny to detect the
    object named in the question.  Frames are scored by their maximum detection
    confidence; the top-budget frames are returned in temporal order.

    Falls back to uniform sampling when no textual object name can be extracted
    from the question (e.g. <BBOX>-only object_motion questions).
    """

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.30
    _BATCH_SIZE = 8

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading GroundingDINO model %s on %s", self.MODEL_ID, self._device
        )
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID
        )
        self._model = self._model.to(self._device).eval()
        logger.info("GroundingDINO loaded.")

    @property
    def name(self) -> str:
        return "object_tracking"

    def _score_frames(self, frames: list[Frame], text_query: str) -> np.ndarray:
        """Return per-frame max detection confidence scores (shape: N,)."""
        scores = np.zeros(len(frames), dtype=np.float32)

        # GroundingDINO expects the query to end with a period
        query = text_query.rstrip(".") + "."

        for start in range(0, len(frames), self._BATCH_SIZE):
            batch = frames[start : start + self._BATCH_SIZE]
            images = [f.image for f in batch]

            inputs = self._processor(
                images=images,
                text=[query] * len(images),
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**inputs)

            # target_sizes: list of (height, width) — PIL .size is (W, H)
            target_sizes = [img.size[::-1] for img in images]
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
                target_sizes=target_sizes,
            )

            for j, result in enumerate(results):
                if len(result["scores"]) > 0:
                    scores[start + j] = float(result["scores"].max().cpu())

        return scores

    def _uniform_fallback(
        self, candidate_frames: list[Frame], budget: int
    ) -> list[Frame]:
        step = len(candidate_frames) / budget
        selected = [candidate_frames[int(i * step)] for i in range(budget)]
        return sorted(selected, key=lambda f: f.timestamp_s)

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

        query = _extract_object_query(question)
        if query is None:
            logger.debug(
                "ObjectTrackingTool: no text query extractable, using uniform fallback"
            )
            return self._uniform_fallback(candidate_frames, budget)

        logger.debug("ObjectTrackingTool: grounding query = %r", query)
        scores = self._score_frames(candidate_frames, query)

        n_detected = int((scores > 0).sum())
        logger.debug(
            "ObjectTrackingTool: %d/%d frames with detections, max_score=%.3f",
            n_detected, len(candidate_frames), float(scores.max()),
        )

        selected = self._temporally_diverse(candidate_frames, scores, budget)
        logger.debug(
            "ObjectTrackingTool: selected %d/%d frames",
            len(selected), len(candidate_frames),
        )
        return selected

    def _temporally_diverse(
        self, frames: list[Frame], scores: np.ndarray, budget: int
    ) -> list[Frame]:
        """Pick the highest-scoring frame per temporal bucket."""
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
            idx = int(np.argmax(scores_t[start:end]))
            selected.append(frames_t[start + idx])
        return selected
