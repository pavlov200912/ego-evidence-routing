"""Local-detail crop refinement evidence tool."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool
from eer.tools.ocr import OCRDetection, OCRTool

logger = logging.getLogger(__name__)

_TEXT_DETAIL_TERMS = {
    "brand",
    "date",
    "display",
    "label",
    "logo",
    "name",
    "number",
    "nutrition",
    "package",
    "packaging",
    "price",
    "read",
    "scale",
    "screen",
    "sign",
    "tag",
    "text",
    "weight",
    "written",
}
_SMALL_OBJECT_TERMS = {
    "ingredient",
    "object",
    "item",
    "tool",
    "utensil",
    "bottle",
    "jar",
    "packet",
    "piece",
}


@dataclass(frozen=True)
class _CropCandidate:
    frame: Frame
    crop_box: tuple[int, int, int, int]
    score: float
    rank_in_frame: int


def _question_tokens(question: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", question.lower()))


def _crop_scale_for_question(question: str) -> float:
    """Choose crop size as a fraction of the shorter image side."""
    tokens = _question_tokens(question)
    if tokens & _TEXT_DETAIL_TERMS:
        return 0.34
    if tokens & _SMALL_OBJECT_TERMS:
        return 0.42
    return 0.50


def _spatial_bias(question: str, height: int, width: int) -> np.ndarray:
    """Return a soft spatial prior for question words such as left/top/center."""
    tokens = _question_tokens(question)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    bias = np.ones((height, width), dtype=np.float32)
    if "left" in tokens:
        bias *= 1.15 - 0.30 * x
    if "right" in tokens:
        bias *= 0.85 + 0.30 * x
    if "top" in tokens or "upper" in tokens:
        bias *= 1.15 - 0.30 * y
    if "bottom" in tokens or "lower" in tokens:
        bias *= 0.85 + 0.30 * y
    if "center" in tokens or "middle" in tokens:
        dist = (x - 0.5) ** 2 + (y - 0.5) ** 2
        bias *= 1.0 + 0.35 * np.exp(-dist / 0.08)

    # Egocentric manipulation evidence is often near the lower-center field of view.
    lower_center = np.exp(-(((x - 0.5) ** 2) / 0.18 + ((y - 0.68) ** 2) / 0.20))
    return bias * (1.0 + 0.15 * lower_center)


def _saliency_map(image: Image.Image, question: str, analysis_size: int) -> np.ndarray:
    """Compute a deterministic local-detail saliency map at thumbnail scale."""
    img = image.convert("RGB").copy()
    img.thumbnail((analysis_size, analysis_size), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    )

    blurred = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2.0)
        ),
        dtype=np.float32,
    ) / 255.0
    local_contrast = np.abs(gray - blurred)

    gy, gx = np.gradient(gray)
    edges = np.hypot(gx, gy)
    saturation = arr.max(axis=2) - arr.min(axis=2)

    tokens = _question_tokens(question)
    if tokens & _TEXT_DETAIL_TERMS:
        saliency = 0.65 * edges + 0.30 * local_contrast + 0.05 * saturation
    else:
        saliency = 0.45 * edges + 0.35 * local_contrast + 0.20 * saturation

    bias = _spatial_bias(question, *saliency.shape)
    saliency = saliency * bias + 0.03 * bias
    max_value = float(saliency.max())
    if max_value > 0:
        saliency = saliency / max_value
    return saliency.astype(np.float32)


def _window_score(saliency: np.ndarray, left: int, top: int, size: int) -> float:
    patch = saliency[top : top + size, left : left + size]
    if patch.size == 0:
        return 0.0
    return float(0.75 * patch.mean() + 0.25 * patch.max())


def _best_crops_for_frame(
    frame: Frame,
    question: str,
    *,
    crops_per_frame: int,
    analysis_size: int,
) -> list[_CropCandidate]:
    """Find the highest-scoring square crop windows for one frame."""
    saliency = _saliency_map(frame.image, question, analysis_size)
    h, w = saliency.shape
    crop_size = max(8, int(round(min(h, w) * _crop_scale_for_question(question))))
    stride = max(1, crop_size // 3)

    scored: list[tuple[float, int, int]] = []
    for top in range(0, max(1, h - crop_size + 1), stride):
        for left in range(0, max(1, w - crop_size + 1), stride):
            scored.append((_window_score(saliency, left, top, crop_size), left, top))

    if not scored:
        return []

    scored.sort(reverse=True)
    selected: list[tuple[float, int, int]] = []
    min_distance = crop_size * 0.55
    for score, left, top in scored:
        cx = left + crop_size / 2.0
        cy = top + crop_size / 2.0
        if all(
            (cx - (other_left + crop_size / 2.0)) ** 2
            + (cy - (other_top + crop_size / 2.0)) ** 2
            >= min_distance**2
            for _, other_left, other_top in selected
        ):
            selected.append((score, left, top))
        if len(selected) >= crops_per_frame:
            break

    orig_w, orig_h = frame.image.size
    scale_x = orig_w / w
    scale_y = orig_h / h
    candidates: list[_CropCandidate] = []
    for rank, (score, left, top) in enumerate(selected):
        x0 = max(0, int(round(left * scale_x)))
        y0 = max(0, int(round(top * scale_y)))
        x1 = min(orig_w, int(round((left + crop_size) * scale_x)))
        y1 = min(orig_h, int(round((top + crop_size) * scale_y)))
        candidates.append(
            _CropCandidate(
                frame=frame,
                crop_box=(x0, y0, x1, y1),
                score=score,
                rank_in_frame=rank,
            )
        )
    return candidates


def _expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding_ratio: float = 0.45,
    min_side: int = 96,
) -> tuple[int, int, int, int]:
    """Pad a detection box while keeping it inside the image."""
    x0, y0, x1, y1 = box
    width, height = image_size
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    side = max(min_side, box_w, box_h)
    pad_x = max(side - box_w, int(round(box_w * padding_ratio))) / 2.0
    pad_y = max(side - box_h, int(round(box_h * padding_ratio))) / 2.0

    return (
        max(0, int(round(x0 - pad_x))),
        max(0, int(round(y0 - pad_y))),
        min(width, int(round(x1 + pad_x))),
        min(height, int(round(y1 + pad_y))),
    )


def _ocr_crops_for_frame(
    frame: Frame,
    detections: list[OCRDetection],
    *,
    min_crop_side: int,
) -> list[_CropCandidate]:
    """Convert OCR text boxes into crop candidates."""
    candidates: list[_CropCandidate] = []
    image_area = max(1, frame.image.width * frame.image.height)
    for rank, detection in enumerate(detections):
        x0, y0, x1, y1 = detection.bbox
        if x1 <= x0 or y1 <= y0:
            continue

        crop_box = _expand_box(
            detection.bbox,
            frame.image.size,
            min_side=min_crop_side,
        )
        area = max(1, (x1 - x0) * (y1 - y0))
        area_score = min(1.0, area / image_area * 40.0)
        text_score = min(1.0, len(detection.text.strip()) / 12.0)
        score = float(2.0 + detection.confidence + 0.25 * area_score + 0.15 * text_score)
        candidates.append(
            _CropCandidate(
                frame=frame,
                crop_box=crop_box,
                score=score,
                rank_in_frame=rank,
            )
        )
    return candidates


def _zoom_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    min_output_side: int,
) -> Image.Image:
    """Crop at native resolution and upsample small regions for VLM inspection."""
    crop = image.convert("RGB").crop(box)
    short_side = min(crop.size)
    if short_side >= min_output_side:
        return crop

    scale = min_output_side / max(1, short_side)
    new_size = (
        max(1, int(round(crop.width * scale))),
        max(1, int(round(crop.height * scale))),
    )
    return crop.resize(new_size, Image.Resampling.LANCZOS)


class CropTool(EvidenceTool):
    """Extract high-resolution local crops from salient regions.

    The tool is detector-free and deterministic: it scores local windows by
    edge density, local contrast, and color variation, then returns zoomed crops
    from the highest-scoring frame regions. Text/detail questions use smaller,
    edge-heavy crops; generic object questions use slightly wider crops.
    """

    def __init__(
        self,
        *,
        crops_per_frame: int = 2,
        analysis_size: int = 224,
        min_output_side: int = 384,
    ) -> None:
        self.crops_per_frame = crops_per_frame
        self.analysis_size = analysis_size
        self.min_output_side = min_output_side

    @property
    def name(self) -> str:
        return "crop"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Return up to *budget* cropped Frame objects in temporal order."""
        if not candidate_frames or budget <= 0:
            return []

        all_candidates: list[_CropCandidate] = []
        for frame in candidate_frames:
            all_candidates.extend(
                _best_crops_for_frame(
                    frame,
                    question,
                    crops_per_frame=max(1, self.crops_per_frame),
                    analysis_size=self.analysis_size,
                )
            )

        if not all_candidates:
            return []

        best = sorted(all_candidates, key=lambda c: c.score, reverse=True)[:budget]
        selected: list[Frame] = []
        for candidate in sorted(
            best,
            key=lambda c: (c.frame.timestamp_s, c.rank_in_frame),
        ):
            crop = _zoom_crop(
                candidate.frame.image,
                candidate.crop_box,
                min_output_side=self.min_output_side,
            )
            selected.append(
                Frame(
                    index=candidate.frame.index * 100 + candidate.rank_in_frame,
                    timestamp_s=candidate.frame.timestamp_s,
                    image=crop,
                )
            )

        logger.debug(
            "CropTool: selected %d crops from %d frames",
            len(selected),
            len(candidate_frames),
        )
        return selected


class OCRCropTool(CropTool):
    """Crop around OCR text boxes, falling back to saliency crops.

    This combines the separate OCR evidence tool with high-resolution crop
    refinement for questions where small scene text, labels, displays, or scale
    readings may contain the answer.
    """

    def __init__(
        self,
        *,
        ocr_tool: OCRTool | None = None,
        ocr_device: str | None = None,
        crops_per_frame: int = 2,
        analysis_size: int = 224,
        min_output_side: int = 384,
        min_crop_side: int = 96,
    ) -> None:
        super().__init__(
            crops_per_frame=crops_per_frame,
            analysis_size=analysis_size,
            min_output_side=min_output_side,
        )
        self.ocr_tool = ocr_tool or OCRTool(device=ocr_device)
        self.min_crop_side = min_crop_side

    @property
    def name(self) -> str:
        return "ocr_crop"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Return OCR-guided crops, using saliency crops to fill the budget."""
        if not candidate_frames or budget <= 0:
            return []

        all_candidates: list[_CropCandidate] = []
        for frame in candidate_frames:
            detections = self.ocr_tool.detect(frame)
            all_candidates.extend(
                _ocr_crops_for_frame(
                    frame,
                    detections,
                    min_crop_side=self.min_crop_side,
                )
            )

        if len(all_candidates) < budget:
            for frame in candidate_frames:
                all_candidates.extend(
                    _best_crops_for_frame(
                        frame,
                        question,
                        crops_per_frame=max(1, self.crops_per_frame),
                        analysis_size=self.analysis_size,
                    )
                )

        if not all_candidates:
            return []

        best = sorted(all_candidates, key=lambda c: c.score, reverse=True)[:budget]
        selected: list[Frame] = []
        for candidate in sorted(
            best,
            key=lambda c: (c.frame.timestamp_s, c.rank_in_frame),
        ):
            crop = _zoom_crop(
                candidate.frame.image,
                candidate.crop_box,
                min_output_side=self.min_output_side,
            )
            selected.append(
                Frame(
                    index=candidate.frame.index * 100 + candidate.rank_in_frame,
                    timestamp_s=candidate.frame.timestamp_s,
                    image=crop,
                )
            )

        logger.debug(
            "OCRCropTool: selected %d crops from %d frames",
            len(selected),
            len(candidate_frames),
        )
        return selected
