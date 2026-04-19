"""Local-detail crop refinement tool — stub for Week 2."""

from __future__ import annotations

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool


class CropTool(EvidenceTool):
    """Refine evidence by cropping salient local regions from selected frames.

    To be implemented by Christina and Eren in Week 2.
    """

    @property
    def name(self) -> str:
        return "crop"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        raise NotImplementedError("CropTool not yet implemented (Week 2).")
