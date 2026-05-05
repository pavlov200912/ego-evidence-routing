"""Union tool: merge the selections of two independent tools."""

from __future__ import annotations

import logging

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


class UnionTool(EvidenceTool):
    """Return the union of two tools' selections.

    Each tool runs independently with the full budget; their outputs are
    merged and deduplicated by frame index. The result can be up to
    2 × budget frames, giving the VLM both temporal coverage (e.g. uniform)
    and semantic relevance (e.g. CLIP) simultaneously.
    """

    def __init__(self, tool_a: EvidenceTool, tool_b: EvidenceTool) -> None:
        self._a = tool_a
        self._b = tool_b

    @property
    def name(self) -> str:
        return f"{self._a.name}+{self._b.name}"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        **kwargs,
    ) -> list[Frame]:
        frames_a = self._a.select(candidate_frames, question, budget=budget, **kwargs)
        frames_b = self._b.select(candidate_frames, question, budget=budget, **kwargs)

        seen: set[int] = set()
        result: list[Frame] = []
        for f in frames_a + frames_b:
            if f.index not in seen:
                seen.add(f.index)
                result.append(f)

        selected = sorted(result, key=lambda f: f.timestamp_s)
        logger.debug(
            "UnionTool(%s): %d + %d → %d frames",
            self.name, len(frames_a), len(frames_b), len(selected),
        )
        return selected
