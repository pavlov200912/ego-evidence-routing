"""HD-EPIC VQA dataset loader.

Placeholder structure — adapt once Eren provides the real data format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(
    [
        "action_recognition",
        "object_state",
        "spatial_relation",
        "temporal_order",
        "counting",
        "causal",
        "social_interaction",
    ]
)

VALID_ANSWER_LETTERS = frozenset("ABCDE")


@dataclass
class VQAQuestion:
    """A single VQA item from HD-EPIC."""

    question_id: str
    video_id: str
    start_time: float  # seconds
    end_time: float  # seconds
    question: str
    choices: list[str]  # exactly 5 strings, indexed 0-4 (A-E)
    correct_answer: str  # letter A–E
    category: str  # one of VALID_CATEGORIES
    prototype: str  # one of ~30 prototype strings

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    def choice_dict(self) -> dict[str, str]:
        """Map answer letter → choice text."""
        return {letter: text for letter, text in zip("ABCDE", self.choices)}


def _parse_question(raw: dict[str, Any]) -> VQAQuestion:
    """Parse a raw dict (from JSON/CSV) into a VQAQuestion.

    Adjust field names here when the real HD-EPIC format is known.
    """
    return VQAQuestion(
        question_id=str(raw["question_id"]),
        video_id=str(raw["video_id"]),
        start_time=float(raw["start_time"]),
        end_time=float(raw["end_time"]),
        question=str(raw["question"]),
        choices=[str(raw[f"choice_{l}"]) for l in "ABCDE"],
        correct_answer=str(raw["correct_answer"]).upper(),
        category=str(raw["category"]),
        prototype=str(raw["prototype"]),
    )


@dataclass
class HDEPICDataset:
    """Collection of HD-EPIC VQA questions with filtering and splitting helpers."""

    questions: list[VQAQuestion] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "HDEPICDataset":
        """Load from a JSON file.

        Expected format: a list of question objects, or a dict with a
        ``"questions"`` key containing the list.

        Args:
            path: Path to the JSON file.
        """
        path = Path(path)
        logger.info("Loading HD-EPIC VQA questions from %s", path)
        with path.open() as f:
            raw = json.load(f)

        items = raw if isinstance(raw, list) else raw["questions"]
        questions = [_parse_question(item) for item in items]
        logger.info("Loaded %d questions", len(questions))
        return cls(questions=questions)

    @classmethod
    def from_csv(cls, path: str | Path) -> "HDEPICDataset":
        """Load from a CSV file.

        Expected columns: question_id, video_id, start_time, end_time,
        question, choice_A, choice_B, choice_C, choice_D, choice_E,
        correct_answer, category, prototype.

        Args:
            path: Path to the CSV file.
        """
        path = Path(path)
        logger.info("Loading HD-EPIC VQA questions from %s", path)
        df = pd.read_csv(path)
        questions = [_parse_question(row) for row in df.to_dict(orient="records")]
        logger.info("Loaded %d questions", len(questions))
        return cls(questions=questions)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_duration(self, max_seconds: float) -> "HDEPICDataset":
        """Return a new dataset with clips no longer than *max_seconds*.

        Args:
            max_seconds: Maximum clip duration in seconds.
        """
        kept = [q for q in self.questions if q.duration_s <= max_seconds]
        logger.info(
            "filter_by_duration(%.0fs): %d → %d questions",
            max_seconds,
            len(self.questions),
            len(kept),
        )
        return HDEPICDataset(questions=kept)

    def filter_by_categories(self, categories: list[str]) -> "HDEPICDataset":
        """Return a new dataset containing only the specified categories.

        Args:
            categories: List of category strings (e.g. ["action_recognition"]).
        """
        cat_set = set(categories)
        kept = [q for q in self.questions if q.category in cat_set]
        logger.info(
            "filter_by_categories(%s): %d → %d questions",
            categories,
            len(self.questions),
            len(kept),
        )
        return HDEPICDataset(questions=kept)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, val_ratio: float = 0.3) -> tuple["HDEPICDataset", "HDEPICDataset"]:
        """Deterministic train/val split (no shuffle, keeps ordering stable).

        Args:
            val_ratio: Fraction of questions to put in val set.

        Returns:
            (train_dataset, val_dataset) tuple.
        """
        n_val = max(1, int(len(self.questions) * val_ratio))
        train_qs = self.questions[n_val:]
        val_qs = self.questions[:n_val]
        logger.info("Split: %d train / %d val", len(train_qs), len(val_qs))
        return HDEPICDataset(questions=train_qs), HDEPICDataset(questions=val_qs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, idx: int) -> VQAQuestion:
        return self.questions[idx]
