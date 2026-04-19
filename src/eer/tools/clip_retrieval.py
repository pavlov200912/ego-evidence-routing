"""CLIP/SigLIP text-image retrieval tool."""

from __future__ import annotations

import logging

import numpy as np
import torch

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


class CLIPRetrievalTool(EvidenceTool):
    """Select frames most semantically relevant to the question via SigLIP.

    Embeds all candidate frame images and the question text, then returns
    the top-K frames by cosine similarity, sorted by timestamp.
    """

    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP",
        pretrained: str = "webli",
        device: str | None = None,
    ) -> None:
        """Load the open_clip model and preprocessing transforms.

        Args:
            model_name: open_clip architecture name.
            pretrained: Pretrained weights tag.
            device: Torch device string; defaults to CUDA if available.
        """
        import open_clip

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading CLIP model %s (%s) on %s", model_name, pretrained, self._device
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        logger.info("CLIP model loaded.")

    @property
    def name(self) -> str:
        return "clip"

    def _embed_images(self, frames: list[Frame]) -> np.ndarray:
        """Return L2-normalized image embeddings, shape (N, D)."""
        images = torch.stack(
            [self._preprocess(f.image) for f in frames]  # type: ignore[arg-type]
        ).to(self._device)
        with torch.inference_mode():
            feats = self._model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def _embed_text(self, text: str) -> np.ndarray:
        """Return L2-normalized text embedding, shape (1, D)."""
        tokens = self._tokenizer([text]).to(self._device)
        with torch.inference_mode():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Return top-*budget* frames by cosine similarity to *question*.

        Args:
            candidate_frames: All available frames for the clip.
            question: VQA question used to construct the text query.
            budget: Number of frames to return.

        Returns:
            Selected frames sorted by ascending timestamp.
        """
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return list(candidate_frames)

        image_embeds = self._embed_images(candidate_frames)  # (N, D)
        text_embed = self._embed_text(question)  # (1, D)

        similarities = (image_embeds @ text_embed.T).squeeze(-1)  # (N,)
        top_indices = np.argsort(similarities)[::-1][:budget]

        selected = sorted(
            [candidate_frames[i] for i in top_indices],
            key=lambda f: f.timestamp_s,
        )
        logger.debug(
            "CLIPRetrievalTool: selected %d/%d frames", len(selected), len(candidate_frames)
        )
        return selected
