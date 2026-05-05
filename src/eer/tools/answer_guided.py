"""Answer-guided frame selection tool for localization questions.

For questions where each answer choice specifies a time window (e.g.
"from <TIME 00:09:53 video 1> to <TIME 00:09:54 video 1>"), extracts one
representative frame per answer window and returns them all. This gives
Qwen direct visual evidence for each candidate answer rather than hoping
uniform or semantic sampling lands on the right moment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from eer.data.frames import Frame, extract_candidate_frames
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"<TIME\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+video\s+\d+>")


def _to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_window(choice: str) -> tuple[float, float] | None:
    """Extract (start_s, end_s) from a choice containing two TIME tokens."""
    matches = _TIME_RE.findall(choice)
    if len(matches) >= 2:
        t0, t1 = _to_seconds(matches[0]), _to_seconds(matches[1])
        return min(t0, t1), max(t0, t1)
    if len(matches) == 1:
        t = _to_seconds(matches[0])
        return t, t + 1.0  # treat single timestamp as a 1-second window
    return None


class AnswerGuidedTool(EvidenceTool):
    """Extract one frame per answer-choice window and return all of them.

    This lets Qwen directly compare a representative frame from each
    candidate time window, which is the right strategy for action/event
    localization questions whose choices are explicit time intervals.

    Falls back to the candidate_frames pool when choices contain no
    parseable time windows (e.g. text-only choices).
    """

    @property
    def name(self) -> str:
        return "answer_guided"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        *,
        choices: list[str] | None = None,
        video_path: str | None = None,
    ) -> list[Frame]:
        windows = self._parse_choice_windows(choices or [])

        if not windows:
            logger.debug("AnswerGuidedTool: no time windows in choices, returning empty")
            return []

        if video_path is None:
            logger.warning("AnswerGuidedTool: no video_path, falling back to candidate pool")
            return self._from_candidates(candidate_frames, windows)

        frames = self._extract_from_windows(Path(video_path), windows)
        logger.debug(
            "AnswerGuidedTool: extracted %d frames from %d answer windows",
            len(frames), len(windows),
        )
        return frames

    # ------------------------------------------------------------------

    def _parse_choice_windows(
        self, choices: list[str]
    ) -> list[tuple[float, float]]:
        windows = []
        for choice in choices:
            w = _parse_window(choice)
            if w is not None:
                windows.append(w)
        return windows

    def _extract_from_windows(
        self, video_path: Path, windows: list[tuple[float, float]]
    ) -> list[Frame]:
        """Extract the midpoint frame of each answer window."""
        frames: list[Frame] = []
        for start_s, end_s in windows:
            mid = (start_s + end_s) / 2
            # Extract at high fps around the midpoint to guarantee a frame
            extracted = extract_candidate_frames(
                video_path, fps=4.0, start_s=max(0, mid - 0.5), end_s=mid + 0.5
            )
            if extracted:
                # Pick the frame closest to the midpoint
                best = min(extracted, key=lambda f: abs(f.timestamp_s - mid))
                frames.append(best)
        return sorted(frames, key=lambda f: f.timestamp_s)

    def _from_candidates(
        self,
        candidate_frames: list[Frame],
        windows: list[tuple[float, float]],
    ) -> list[Frame]:
        """Fallback: pick the candidate frame closest to each window midpoint."""
        if not candidate_frames:
            return []
        frames: list[Frame] = []
        for start_s, end_s in windows:
            mid = (start_s + end_s) / 2
            best = min(candidate_frames, key=lambda f: abs(f.timestamp_s - mid))
            if best not in frames:
                frames.append(best)
        return sorted(frames, key=lambda f: f.timestamp_s)
