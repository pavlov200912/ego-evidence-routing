"""Cascade tool: two-stage evidence selection."""

from __future__ import annotations

import logging

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


class CascadeTool(EvidenceTool):
    """Two-stage cascade: first tool over-selects, second tool refines.

    Stage 1 runs with budget * overselect_factor to produce a larger
    candidate pool. Stage 2 then selects the final budget from that pool.

    Example: CLIP (stage 1) retrieves semantically relevant frames, then
    SharpnessStability (stage 2) picks the clearest among them.
    """

    def __init__(
        self,
        stage1: EvidenceTool,
        stage2: EvidenceTool,
        overselect_factor: int = 3,
    ) -> None:
        self._stage1 = stage1
        self._stage2 = stage2
        self._overselect_factor = overselect_factor

    @property
    def name(self) -> str:
        return f"{self._stage1.name}_then_{self._stage2.name}"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        if not candidate_frames:
            return []

        pool_size = min(budget * self._overselect_factor, len(candidate_frames))
        pool = self._stage1.select(candidate_frames, question, budget=pool_size)

        selected = self._stage2.select(pool, question, budget=budget)
        logger.debug(
            "CascadeTool(%s): %d → %d → %d frames",
            self.name,
            len(candidate_frames),
            len(pool),
            len(selected),
        )
        return selected
