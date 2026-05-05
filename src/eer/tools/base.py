"""Abstract base class for auxiliary evidence selection tools."""

from __future__ import annotations

from abc import ABC, abstractmethod

from eer.data.frames import Frame


class EvidenceTool(ABC):
    """Base class for auxiliary evidence selection tools.

    Each tool takes the full list of candidate frames (extracted at 1 fps)
    and returns a ranked/selected subset of size *budget*.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables and logs."""
        ...

    @abstractmethod
    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        *,
        choices: list[str] | None = None,
        video_path: str | None = None,
    ) -> list[Frame]:
        """Select *budget* frames from *candidate_frames*.

        Implementations should return frames in ascending temporal order.

        Args:
            candidate_frames: All frames available for this clip.
            question: The VQA question text (may be used for text-guided tools).
            budget: Number of frames to return.
            choices: Answer choices (optional, used by answer-guided tools).
            video_path: Path to the source video (optional, used by tools that
                need to extract frames beyond the candidate pool).

        Returns:
            A list of at most *budget* Frame objects.
        """
        ...
