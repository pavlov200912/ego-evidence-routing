"""Tests for evidence selection tools — use synthetic frames, no GPU needed."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eer.data.frames import Frame
from eer.tools.crop import CropTool, OCRCropTool
from eer.tools.hand import HandTool
from eer.tools.motion import MotionTool
from eer.tools.ocr import OCRDetection
from eer.tools.uniform import UniformTool


def _make_frames(n: int, size: tuple[int, int] = (64, 64)) -> list[Frame]:
    """Create *n* synthetic frames with random pixel data."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n):
        arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        frames.append(Frame(index=i, timestamp_s=float(i), image=img))
    return frames


QUESTION = "What is the person holding?"


class _FakeOCRTool:
    def __init__(self, detections: list[OCRDetection] | None = None) -> None:
        self.detections = detections or []

    def detect(self, frame: Frame) -> list[OCRDetection]:
        return self.detections


# --- UniformTool ---

def test_uniform_returns_budget(tmp_path) -> None:
    frames = _make_frames(20)
    tool = UniformTool()
    selected = tool.select(frames, QUESTION, budget=8)
    assert len(selected) == 8


def test_uniform_fewer_than_budget(tmp_path) -> None:
    frames = _make_frames(5)
    tool = UniformTool()
    selected = tool.select(frames, QUESTION, budget=8)
    assert len(selected) == 5


def test_uniform_temporal_order() -> None:
    frames = _make_frames(20)
    tool = UniformTool()
    selected = tool.select(frames, QUESTION, budget=6)
    timestamps = [f.timestamp_s for f in selected]
    assert timestamps == sorted(timestamps)


def test_uniform_empty_input() -> None:
    tool = UniformTool()
    assert tool.select([], QUESTION, budget=8) == []


def test_uniform_name() -> None:
    assert UniformTool().name == "uniform"


# --- MotionTool ---

def test_motion_returns_budget() -> None:
    frames = _make_frames(30)
    tool = MotionTool()
    selected = tool.select(frames, QUESTION, budget=8)
    assert len(selected) == 8


def test_motion_temporal_order() -> None:
    frames = _make_frames(30)
    tool = MotionTool()
    selected = tool.select(frames, QUESTION, budget=8)
    timestamps = [f.timestamp_s for f in selected]
    assert timestamps == sorted(timestamps)


def test_motion_prefers_high_motion_frames() -> None:
    """Frames with large pixel differences should be preferred."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(20):
        if i in (5, 10, 15):
            # High-motion frame: random noise
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        else:
            # Low-motion: constant gray
            arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        frames.append(Frame(index=i, timestamp_s=float(i), image=Image.fromarray(arr)))

    tool = MotionTool()
    selected = tool.select(frames, QUESTION, budget=3)
    selected_ts = {f.timestamp_s for f in selected}
    # At least 2 of the 3 high-motion frames should be selected
    high_motion_ts = {5.0, 10.0, 15.0}
    assert len(selected_ts & high_motion_ts) >= 2


def test_motion_empty_input() -> None:
    tool = MotionTool()
    assert tool.select([], QUESTION, budget=8) == []


def test_motion_name() -> None:
    assert MotionTool().name == "motion"


# --- CropTool ---

def test_crop_returns_budget() -> None:
    frames = _make_frames(12, size=(96, 128))
    tool = CropTool(analysis_size=64, min_output_side=128)
    selected = tool.select(frames, QUESTION, budget=5)
    assert len(selected) == 5


def test_crop_temporal_order() -> None:
    frames = _make_frames(12, size=(96, 128))
    tool = CropTool(analysis_size=64, min_output_side=128)
    selected = tool.select(frames, QUESTION, budget=5)
    timestamps = [f.timestamp_s for f in selected]
    assert timestamps == sorted(timestamps)


def test_crop_returns_zoomed_crops() -> None:
    frames = _make_frames(3, size=(96, 128))
    tool = CropTool(analysis_size=64, min_output_side=128)
    selected = tool.select(frames, "What number is written on the label?", budget=2)
    assert selected
    assert all(min(f.image.size) >= 128 for f in selected)
    assert all(f.image.size != frames[0].image.size for f in selected)


def test_crop_empty_input() -> None:
    tool = CropTool()
    assert tool.select([], QUESTION, budget=8) == []


def test_crop_name() -> None:
    assert CropTool().name == "crop"


def test_ocr_crop_uses_ocr_boxes() -> None:
    img = Image.new("RGB", (200, 200), "white")
    frame = Frame(index=2, timestamp_s=4.5, image=img)
    detection = OCRDetection(
        bbox=(80, 90, 100, 105),
        text="125g",
        confidence=0.95,
    )
    tool = OCRCropTool(
        ocr_tool=_FakeOCRTool([detection]),  # type: ignore[arg-type]
        min_crop_side=80,
        min_output_side=120,
    )

    selected = tool.select([frame], "What number is written on the label?", budget=1)

    assert len(selected) == 1
    assert selected[0].timestamp_s == frame.timestamp_s
    assert min(selected[0].image.size) >= 120
    assert selected[0].image.size != frame.image.size


def test_ocr_crop_falls_back_to_saliency() -> None:
    frames = _make_frames(4, size=(96, 128))
    tool = OCRCropTool(
        ocr_tool=_FakeOCRTool(),  # type: ignore[arg-type]
        analysis_size=64,
        min_output_side=128,
    )

    selected = tool.select(frames, QUESTION, budget=2)

    assert len(selected) == 2
    assert all(min(f.image.size) >= 128 for f in selected)


# --- Stub tools raise NotImplementedError ---

def test_hand_tool_raises() -> None:
    frames = _make_frames(5)
    with pytest.raises(NotImplementedError):
        HandTool().select(frames, QUESTION)
