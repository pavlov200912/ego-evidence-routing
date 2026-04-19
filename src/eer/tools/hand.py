"""Hand / object interaction evidence tool — stub for Week 2."""

from __future__ import annotations

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool


class HandTool(EvidenceTool):
    """Select frames showing prominent hand-object interactions.

    To be implemented by Christina and Eren in Week 2.
    """

    @property
    def name(self) -> str:
        return "hand"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        raise NotImplementedError("HandTool not yet implemented (Week 2).")
