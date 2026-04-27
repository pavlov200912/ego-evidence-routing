"""OCR / Local-detail crop evidence tool."""

from __future__ import annotations

import logging

from PIL import Image

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)

class OCRTool(EvidenceTool):
    """Detect and extract text from frames using an OCR model."""

    def __init__(self, device: str = "cpu"):
        """Initialize OCR model (e.g. EasyOCR)."""
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "Please install easyocr to use the OCRTool: `pip install easyocr`"
            )
        
        self.device = device
        # Use simple English OCR for now
        self.reader = easyocr.Reader(['en'], gpu=(device != "cpu"))
        logger.info("EasyOCR model loaded.")

    @property
    def name(self) -> str:
        return "ocr"

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
        import numpy as np

        if not candidate_frames:
            return []
            
        if len(candidate_frames) <= budget:
            return list(candidate_frames)
            
        scores = []
        for i, frame in enumerate(candidate_frames):
            # EasyOCR expects numpy array or path
            img_np = np.array(frame.image)
            results = self.reader.readtext(img_np)
            
            # Simple heuristic: score is total confidence of all detected text boxes
            score = sum([res[2] for res in results]) if results else 0.0
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