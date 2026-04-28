"""OCR / Local-detail crop evidence tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRDetection:
    """A text detection returned by OCR."""

    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


def _easyocr_bbox_to_xyxy(bbox: list[list[float]] | np.ndarray) -> tuple[int, int, int, int]:
    """Convert EasyOCR's quadrilateral box to integer ``(x0, y0, x1, y1)``."""
    points = np.asarray(bbox, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max())),
        int(np.ceil(ys.max())),
    )


def _default_ocr_device() -> str:
    """Use CUDA for EasyOCR when PyTorch can see a GPU."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class OCRTool(EvidenceTool):
    """Detect and extract text from frames using an OCR model."""

    def __init__(self, device: str | None = None):
        """Initialize OCR model (e.g. EasyOCR)."""
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "Please install easyocr to use the OCRTool: `pip install easyocr`"
            )
        
        self.device = device or _default_ocr_device()
        # Use simple English OCR for now
        self.reader = easyocr.Reader(['en'], gpu=(self.device != "cpu"))
        logger.info("EasyOCR model loaded on %s.", self.device)

    @property
    def name(self) -> str:
        return "ocr"

    def detect(self, frame: Frame) -> list[OCRDetection]:
        """Return OCR text boxes for one frame."""
        img_np = np.array(frame.image)
        results = self.reader.readtext(img_np)
        detections: list[OCRDetection] = []
        for bbox, text, confidence in results:
            detections.append(
                OCRDetection(
                    bbox=_easyocr_bbox_to_xyxy(bbox),
                    text=str(text),
                    confidence=float(confidence),
                )
            )
        return detections

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Rank frames based on the amount/confidence of text detected in them.
        
        Args:
            candidate_frames: All available frames for the clip.
            question: VQA question (could be used to match specific text).
            budget: Number of frames to return.
            
        Returns:
            Selected frames containing the most text, sorted chronologically.
        """
        if not candidate_frames:
            return []
            
        if len(candidate_frames) <= budget:
            return list(candidate_frames)
            
        scores = []
        for i, frame in enumerate(candidate_frames):
            # Simple heuristic: score is total confidence of all detected text boxes
            detections = self.detect(frame)
            score = sum(d.confidence for d in detections)
            scores.append((i, score))
            
        # Sort by highest score (most confident text)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top budget indices
        top_indices = [idx for idx, score in scores[:budget]]
        
        # Sort chronologically
        selected = sorted(
            [candidate_frames[i] for i in top_indices],
            key=lambda f: f.timestamp_s
        )
        
        logger.debug(f"OCRTool: selected {len(selected)}/{len(candidate_frames)} frames")
        return selected
