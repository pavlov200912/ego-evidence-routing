"""Tests for evidence selection tools — use synthetic frames, no GPU needed."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eer.data.frames import Frame
from eer.tools.crop import CropTool
from eer.tools.hand import HandTool
from eer.tools.motion import MotionTool
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


# --- Stub tools raise NotImplementedError ---

def test_hand_tool_raises() -> None:
    frames = _make_frames(5)
    with pytest.raises(NotImplementedError):
        HandTool().select(frames, QUESTION)


def test_crop_tool_raises() -> None:
    frames = _make_frames(5)
    with pytest.raises(NotImplementedError):
        CropTool().select(frames, QUESTION)
