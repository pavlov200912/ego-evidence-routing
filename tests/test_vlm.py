"""Tests for VLM inference helpers — no GPU or model download required."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from eer.data.frames import Frame
from eer.vlm.qwen import (
    VLMResult,
    _append_auxiliary_frame_content,
    _build_choice_prompt,
    _clear_sampling_defaults,
    _extract_letter,
    _extract_log_probs,
    _format_timestamp,
)


CHOICES = ["Cooking", "Reading", "Running", "Sleeping", "Writing"]
QUESTION = "What is the person doing?"


# --- Prompt builder ---

def test_build_choice_prompt_contains_question() -> None:
    prompt = _build_choice_prompt(QUESTION, CHOICES)
    assert QUESTION in prompt


def test_build_choice_prompt_contains_all_letters() -> None:
    prompt = _build_choice_prompt(QUESTION, CHOICES)
    for letter in "ABCDE":
        assert f"{letter}." in prompt


def test_build_choice_prompt_ends_with_instruction() -> None:
    prompt = _build_choice_prompt(QUESTION, CHOICES)
    assert "Answer with the letter only." in prompt


# --- Letter extractor ---

@pytest.mark.parametrize("raw,expected", [
    ("A", "A"),
    ("B.", "B"),
    ("The answer is C.", "C"),
    ("  d  ", "D"),  # case-insensitive
    ("E\n", "E"),
    ("unknown", "?"),
    ("", "?"),
])
def test_extract_letter(raw: str, expected: str) -> None:
    assert _extract_letter(raw) == expected


# --- Log-prob extractor ---

def _make_scores(vocab_size: int = 100) -> tuple[torch.Tensor, ...]:
    """Single-step scores with a spike at token id 65 ('A' in ASCII)."""
    logits = torch.zeros(1, vocab_size)
    logits[0, 65] = 10.0  # strong preference for token 65
    return (logits,)


def test_extract_log_probs_returns_all_letters() -> None:
    tokenizer = MagicMock()
    # Map each letter to a distinct token id
    letter_to_id = {"A": 65, "B": 66, "C": 67, "D": 68, "E": 69}
    tokenizer.encode = lambda text, **_: [letter_to_id[text]] if text in letter_to_id else []

    scores = _make_scores(vocab_size=200)
    log_probs = _extract_log_probs(scores, tokenizer)

    assert set(log_probs.keys()) == set("ABCDE")
    for lp in log_probs.values():
        assert math.isfinite(lp)


def test_extract_log_probs_highest_for_spiked_token() -> None:
    tokenizer = MagicMock()
    letter_to_id = {"A": 65, "B": 66, "C": 67, "D": 68, "E": 69}
    tokenizer.encode = lambda text, **_: [letter_to_id[text]] if text in letter_to_id else []

    scores = _make_scores(vocab_size=200)
    log_probs = _extract_log_probs(scores, tokenizer)

    # Token 65 → "A" should have the highest log-prob
    assert log_probs["A"] == max(log_probs.values())


def test_extract_log_probs_empty_scores() -> None:
    tokenizer = MagicMock()
    log_probs = _extract_log_probs((), tokenizer)
    assert set(log_probs.keys()) == set("ABCDE")
    assert all(v == float("-inf") for v in log_probs.values())


def test_clear_sampling_defaults_removes_sampling_only_flags() -> None:
    config = MagicMock()
    config.temperature = 0.7
    config.top_p = 0.8
    config.top_k = 20

    _clear_sampling_defaults(config)

    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None


def test_format_timestamp() -> None:
    assert _format_timestamp(12.34) == "12.3s"
    assert _format_timestamp(75.0) == "1m 15.0s"


def test_append_auxiliary_frame_content_includes_timestamp() -> None:
    img = Image.new("RGB", (8, 8), "white")
    frame = Frame(index=3, timestamp_s=75.0, image=img)
    content: list[dict] = []

    _append_auxiliary_frame_content(content, [frame])

    assert content[0]["type"] == "text"
    assert "timestamp 1m 15.0s" in content[0]["text"]
    assert content[1] == {"type": "image", "image": img}


# --- VLMResult dataclass ---

def test_vlm_result_fields() -> None:
    r = VLMResult(
        predicted_letter="C",
        log_probs={"A": -1.0, "B": -2.0, "C": -0.5, "D": -3.0, "E": -4.0},
        raw_output="C",
    )
    assert r.predicted_letter == "C"
    assert r.log_probs["C"] == -0.5
    assert r.raw_output == "C"


# --- QwenVLM construction (mocked, no real model) ---

def test_qwen_vlm_init_calls_from_pretrained() -> None:
    with (
        patch("eer.vlm.qwen.Qwen3VLForConditionalGeneration") as mock_model_cls,  # type: ignore[attr-defined]
        patch("eer.vlm.qwen.AutoProcessor") as mock_proc_cls,
    ):
        # Prevent torch from crashing on device ops
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_model.generation_config.temperature = 0.7
        mock_model.generation_config.top_p = 0.8
        mock_model.generation_config.top_k = 20
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        from eer.vlm.qwen import QwenVLM

        vlm = QwenVLM(model_name="fake/model", dtype="float32", device_map="cpu")

        mock_model_cls.from_pretrained.assert_called_once()
        mock_proc_cls.from_pretrained.assert_called_once_with("fake/model")
        assert vlm.model.generation_config.temperature is None
        assert vlm.model.generation_config.top_p is None
        assert vlm.model.generation_config.top_k is None
