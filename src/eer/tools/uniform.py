"""Uniform frame sampling baseline tool."""

from __future__ import annotations

import logging

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


class UniformTool(EvidenceTool):
    """Evenly-spaced frame selection — the simplest possible baseline.

    Does not use the question text; purely time-based.
    """

    @property
    def name(self) -> str:
        return "uniform"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Return *budget* evenly-spaced frames from *candidate_frames*.

        Args:
            candidate_frames: All available frames for the clip.
            question: Not used by this tool.
            budget: Number of frames to select.

        Returns:
            Selected frames in temporal order.
        """
        if not candidate_frames:
            return []

        n = len(candidate_frames)
        if n <= budget:
            return list(candidate_frames)

        # Choose evenly-spaced indices across [0, n-1]
        step = n / budget
        indices = [int(i * step) for i in range(budget)]
        selected = [candidate_frames[i] for i in indices]
        logger.debug("UniformTool: selected %d/%d frames", len(selected), n)
        return selected
