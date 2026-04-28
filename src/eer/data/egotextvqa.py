"""EgoTextVQA dataset loader.

Annotation format (JSONL, one JSON object per line):
    {
        "video_id": "<uuid>",
        "question_id": "<str>",
        "question_type": "<str>",       # e.g. "shopping"
        "question": "<str>",
        "correct_answer": "<str>",
        "timestamp": <float>,           # seconds into the video
        "video_url": "<str>"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EgoTextVQAQuestion:
    """A single open-ended VQA item from EgoTextVQA."""

    question_id: str
    video_id: str
    question_type: str
    question: str
    correct_answer: str
    timestamp: float   # seconds
    video_url: str


def _parse_question(raw: dict[str, Any]) -> EgoTextVQAQuestion:
    return EgoTextVQAQuestion(
        question_id=str(raw["question_id"]),
        video_id=str(raw["video_id"]),
        question_type=str(raw["question_type"]),
        question=str(raw["question"]),
        correct_answer=str(raw["correct_answer"]),
        timestamp=float(raw["timestamp"]),
        video_url=str(raw.get("video_url", "")),
    )


@dataclass
class EgoTextVQADataset:
    """Collection of EgoTextVQA questions with filtering helpers."""

    questions: list[EgoTextVQAQuestion] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EgoTextVQADataset":
        """Load from a JSONL file (one JSON object per line).

        Args:
            path: Path to the .jsonl annotation file.
        """
        path = Path(path)
        logger.info("Loading EgoTextVQA questions from %s", path)
        with path.open() as f:
            questions = [_parse_question(json.loads(line)) for line in f if line.strip()]
        logger.info("Loaded %d questions", len(questions))
        return cls(questions=questions)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_question_type(self, question_types: list[str]) -> "EgoTextVQADataset":
        """Return a new dataset containing only the specified question types.

        Args:
            question_types: List of question type strings (e.g. ["shopping"]).
        """
        type_set = set(question_types)
        kept = [q for q in self.questions if q.question_type in type_set]
        logger.info(
            "filter_by_question_type(%s): %d → %d questions",
            question_types,
            len(self.questions),
            len(kept),
        )
        return EgoTextVQADataset(questions=kept)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, val_ratio: float = 0.3) -> tuple["EgoTextVQADataset", "EgoTextVQADataset"]:
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
        return EgoTextVQADataset(questions=train_qs), EgoTextVQADataset(questions=val_qs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, idx: int) -> EgoTextVQAQuestion:
        return self.questions[idx]
