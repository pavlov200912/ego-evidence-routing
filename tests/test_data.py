"""Tests for dataset loading and filtering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eer.data.hdepic import HDEPICDataset, VQAQuestion


def _make_question(qid: str = "q1", duration: float = 30.0, category: str = "action_recognition", prototype: str = "proto_1") -> dict:
    return {
        "question_id": qid,
        "video_id": "vid_001",
        "start_time": 0.0,
        "end_time": duration,
        "question": "What is the person doing?",
        "choice_A": "Cooking",
        "choice_B": "Reading",
        "choice_C": "Running",
        "choice_D": "Sleeping",
        "choice_E": "Writing",
        "correct_answer": "A",
        "category": category,
        "prototype": prototype,
    }


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    data = [
        _make_question("q1", duration=30.0, category="action_recognition"),
        _make_question("q2", duration=120.0, category="counting"),
        _make_question("q3", duration=700.0, category="action_recognition"),  # too long
    ]
    p = tmp_path / "vqa.json"
    p.write_text(json.dumps(data))
    return p


def test_from_json_loads_all(sample_json: Path) -> None:
    ds = HDEPICDataset.from_json(sample_json)
    assert len(ds) == 3


def test_from_json_question_fields(sample_json: Path) -> None:
    ds = HDEPICDataset.from_json(sample_json)
    q = ds[0]
    assert isinstance(q, VQAQuestion)
    assert q.question_id == "q1"
    assert len(q.choices) == 5
    assert q.correct_answer == "A"


def test_filter_by_duration(sample_json: Path) -> None:
    ds = HDEPICDataset.from_json(sample_json).filter_by_duration(600.0)
    assert len(ds) == 2
    assert all(q.duration_s <= 600.0 for q in ds)


def test_filter_by_categories(sample_json: Path) -> None:
    ds = HDEPICDataset.from_json(sample_json).filter_by_categories(["counting"])
    assert len(ds) == 1
    assert ds[0].category == "counting"


def test_split(sample_json: Path) -> None:
    ds = HDEPICDataset.from_json(sample_json)
    train, val = ds.split(val_ratio=0.33)
    assert len(train) + len(val) == len(ds)
    assert len(val) >= 1


def test_choice_dict(sample_json: Path) -> None:
    q = HDEPICDataset.from_json(sample_json)[0]
    cd = q.choice_dict()
    assert set(cd.keys()) == set("ABCDE")
    assert cd["A"] == "Cooking"


def test_from_csv(tmp_path: Path) -> None:
    import pandas as pd

    rows = [_make_question(f"q{i}") for i in range(5)]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "vqa.csv"
    df.to_csv(csv_path, index=False)

    ds = HDEPICDataset.from_csv(csv_path)
    assert len(ds) == 5
