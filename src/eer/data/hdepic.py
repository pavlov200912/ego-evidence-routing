"""HD-EPIC VQA dataset loader.

Loads questions from the vqa-benchmark/ directory of hd-epic-annotations.
Each JSON file in that directory corresponds to one task/category.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIME_TOKEN_RE = re.compile(r"<TIME\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+video\s+1>")


def _hhmmss_to_seconds(t: str) -> float:
    """Convert 'HH:MM:SS.mmm' to total seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _extract_question_time_window(question: str) -> tuple[float, float] | None:
    """Extract (start_s, end_s) from a question containing two <TIME> tokens.

    Returns None if the question contains anything other than exactly two TIME
    tokens referencing video 1 (e.g. localization questions where times are in
    the answer choices, not the question body).
    """
    matches = _TIME_TOKEN_RE.findall(question)
    if len(matches) == 2:
        t0, t1 = _hhmmss_to_seconds(matches[0]), _hhmmss_to_seconds(matches[1])
        return (min(t0, t1), max(t0, t1))
    return None


@dataclass
class VQAQuestion:
    """A single VQA item from the HD-EPIC benchmark."""

    question_id: str       # e.g. "fine_grained_action_localization_0"
    video_id: str          # e.g. "P01-20240204-121042"
    start_time: float | None  # clip window start in seconds (None = whole video)
    end_time: float | None    # clip window end in seconds (None = whole video)
    question: str
    choices: list[str]     # always 5 strings
    correct_idx: int       # 0-indexed (0–4)
    category: str          # filename stem, e.g. "fine_grained_action_localization"

    @property
    def correct_answer(self) -> str:
        """Correct choice as a letter A–E."""
        return "ABCDE"[self.correct_idx]

    @property
    def duration_s(self) -> float | None:
        """Clip window length in seconds, or None when the full video is used."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def choice_dict(self) -> dict[str, str]:
        return {letter: text for letter, text in zip("ABCDE", self.choices)}


def _parse_entry(question_id: str, raw: dict[str, Any], category: str) -> VQAQuestion:
    vid = raw["inputs"]["video 1"]
    start = _hhmmss_to_seconds(vid["start_time"]) if "start_time" in vid else None
    end = _hhmmss_to_seconds(vid["end_time"]) if "end_time" in vid else None

    # Fall back to parsing the question text when the JSON has no clip window.
    # This covers retrieval-style questions where the temporal context is
    # expressed as "<TIME HH:MM:SS video 1>" tokens inside the question.
    if start is None and end is None:
        parsed = _extract_question_time_window(raw["question"])
        if parsed is not None:
            start, end = parsed

    return VQAQuestion(
        question_id=question_id,
        video_id=vid["id"],
        start_time=start,
        end_time=end,
        question=raw["question"],
        choices=list(raw["choices"]),
        correct_idx=int(raw["correct_idx"]),
        category=category,
    )


@dataclass
class HDEPICDataset:
    """Collection of HD-EPIC VQA questions with filtering helpers."""

    questions: list[VQAQuestion] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "HDEPICDataset":
        """Load one category JSON file from vqa-benchmark/.

        The category name is taken from the filename stem
        (e.g. fine_grained_action_localization.json → 'fine_grained_action_localization').
        """
        path = Path(path)
        category = path.stem
        logger.info("Loading %s", path)
        raw: dict[str, Any] = json.loads(path.read_text())
        questions = [_parse_entry(qid, entry, category) for qid, entry in raw.items()]
        logger.info("Loaded %d questions (category=%s)", len(questions), category)
        return cls(questions=questions)

    @classmethod
    def from_dir(
        cls,
        vqa_dir: str | Path,
        categories: list[str] | None = None,
    ) -> "HDEPICDataset":
        """Load all (or selected) JSON files from a vqa-benchmark/ directory.

        Args:
            vqa_dir: Path to the vqa-benchmark/ directory.
            categories: If given, only load files whose stem is in this list.
                        E.g. ['fine_grained_action_localization', 'recipe_step_localization']
        """
        vqa_dir = Path(vqa_dir)
        all_questions: list[VQAQuestion] = []
        for json_file in sorted(vqa_dir.glob("*.json")):
            if categories is not None and json_file.stem not in categories:
                continue
            sub = cls.from_json(json_file)
            all_questions.extend(sub.questions)
        logger.info("Loaded %d questions total from %s", len(all_questions), vqa_dir)
        return cls(questions=all_questions)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_duration(self, max_seconds: float) -> "HDEPICDataset":
        """Keep questions whose clip window is at most max_seconds.

        Questions with no clip window (whole video) are always kept.
        """
        kept = [
            q for q in self.questions
            if q.duration_s is None or q.duration_s <= max_seconds
        ]
        logger.info(
            "filter_by_duration(%.0fs): %d → %d questions",
            max_seconds, len(self.questions), len(kept),
        )
        return HDEPICDataset(questions=kept)

    def filter_by_min_duration(self, min_seconds: float) -> "HDEPICDataset":
        """Keep questions whose clip window is at least min_seconds.

        Questions with no clip window (whole video) are always kept.
        """
        kept = [
            q for q in self.questions
            if q.duration_s is None or q.duration_s >= min_seconds
        ]
        logger.info(
            "filter_by_min_duration(%.0fs): %d → %d questions",
            min_seconds, len(self.questions), len(kept),
        )
        return HDEPICDataset(questions=kept)

    def filter_by_categories(self, categories: list[str]) -> "HDEPICDataset":
        """Keep only questions whose category is in *categories*."""
        cat_set = set(categories)
        kept = [q for q in self.questions if q.category in cat_set]
        logger.info(
            "filter_by_categories(%s): %d → %d questions",
            categories, len(self.questions), len(kept),
        )
        return HDEPICDataset(questions=kept)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, val_ratio: float = 0.3) -> tuple["HDEPICDataset", "HDEPICDataset"]:
        """Deterministic train/val split (first val_ratio fraction = val)."""
        n_val = max(1, int(len(self.questions) * val_ratio))
        return (
            HDEPICDataset(questions=self.questions[n_val:]),
            HDEPICDataset(questions=self.questions[:n_val]),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, idx: int) -> VQAQuestion:
        return self.questions[idx]
