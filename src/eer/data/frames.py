"""Frame extraction from video clips using decord, with disk caching."""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(".frame_cache")


@dataclass
class Frame:
    """A single decoded video frame."""

    index: int  # 0-based position in the candidate frame list
    timestamp_s: float  # time within the original video
    image: Image.Image


def _cache_dir(video_path: Path, fps: float) -> Path:
    """Deterministic cache directory based on video path and fps."""
    key = hashlib.md5(f"{video_path.resolve()}@{fps}".encode()).hexdigest()[:12]
    return _CACHE_ROOT / key


def _manifest_path(cache: Path) -> Path:
    return cache / "manifest.csv"


def _load_from_cache(cache: Path) -> list[Frame] | None:
    """Return frames from cache, or None if the cache is incomplete."""
    manifest = _manifest_path(cache)
    if not manifest.exists():
        return None
    frames: list[Frame] = []
    with manifest.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = cache / row["filename"]
            if not img_path.exists():
                return None
            frames.append(
                Frame(
                    index=int(row["index"]),
                    timestamp_s=float(row["timestamp_s"]),
                    image=Image.open(img_path).convert("RGB"),
                )
            )
    logger.debug("Loaded %d frames from cache %s", len(frames), cache)
    return frames


def _save_to_cache(cache: Path, frames: list[Frame]) -> None:
    """Write frames to disk and update the manifest."""
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for frame in frames:
        filename = f"{frame.index:06d}.jpg"
        frame.image.save(cache / filename, format="JPEG", quality=95)
        rows.append({"index": frame.index, "timestamp_s": frame.timestamp_s, "filename": filename})
    with _manifest_path(cache).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "timestamp_s", "filename"])
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Cached %d frames to %s", len(frames), cache)


def extract_candidate_frames(
    video_path: str | Path,
    fps: float = 1.0,
    cache_dir: Path | None = None,
) -> list[Frame]:
    """Extract frames at *fps* from *video_path* using decord.

    Results are cached to disk as JPEG files alongside a manifest CSV.
    On subsequent calls the cache is returned without re-decoding.

    Args:
        video_path: Path to the video file.
        fps: Sampling rate in frames per second.
        cache_dir: Override default cache root (useful in tests).

    Returns:
        List of Frame objects in temporal order.  Empty list on failure.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        logger.warning("Video file not found: %s — returning empty frame list", video_path)
        return []

    cache = (cache_dir or _CACHE_ROOT) / _cache_dir(video_path, fps).name
    cached = _load_from_cache(cache)
    if cached is not None:
        return cached

    try:
        import decord  # local import so the rest of the module works without it

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps()
        total_frames = len(vr)
        step = max(1, int(round(native_fps / fps)))

        frame_indices = list(range(0, total_frames, step))
        if not frame_indices:
            return []

        raw_frames = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, C) uint8

        frames: list[Frame] = []
        for i, (fi, arr) in enumerate(zip(frame_indices, raw_frames)):
            timestamp = fi / native_fps
            img = Image.fromarray(arr.astype(np.uint8))
            frames.append(Frame(index=i, timestamp_s=timestamp, image=img))

        _save_to_cache(cache, frames)
        logger.info("Extracted %d frames from %s at %.1f fps", len(frames), video_path, fps)
        return frames

    except Exception:
        logger.exception("Failed to extract frames from %s", video_path)
        return []
