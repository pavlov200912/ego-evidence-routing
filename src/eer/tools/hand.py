"""Hand-object interaction evidence tool using hands23 detector."""

from __future__ import annotations

import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# hands23 detector paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HANDS23_DIR = _PROJECT_ROOT / "hands23_detector"
_HANDS23_CONFIG = _PROJECT_ROOT / "faster_rcnn_X.yaml"
_HANDS23_BASE_CONFIG = _PROJECT_ROOT / "Base-RCNN-FPN.yaml"

_WEIGHTS_URL = (
    "https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth"
)
_DEFAULT_CACHE_DIR = Path(os.environ.get("EER_CACHE_DIR", str(Path.home() / ".cache" / "eer")))
_WEIGHTS_FILENAME = "model_hands23.pth"

# Contact-state weights for scoring
_CONTACT_WEIGHT = {
    "object_contact": 1.0,      # portable object contact
    "obj_to_obj_contact": 0.6,  # stationary object contact
    "self_contact": 0.2,
    "other_person_contact": 0.0,
    "no_contact": 0.0,
}

# Singleton predictor cache
_predictor_cache: dict[str, object] = {}


def _ensure_weights(cache_dir: Path | None = None) -> Path:
    """Download hands23 weights if not already cached. Returns path to .pth file."""
    cache = cache_dir or _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    weights_path = cache / _WEIGHTS_FILENAME
    if weights_path.exists():
        logger.debug("hands23 weights found at %s", weights_path)
        return weights_path
    logger.info("Downloading hands23 weights to %s ...", weights_path)
    import ssl
    import certifi
    import urllib.request

    ctx = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(_WEIGHTS_URL, context=ctx) as r:
        with open(weights_path, "wb") as f:
            f.write(r.read())
    logger.info("Download complete.")
    return weights_path


def _get_predictor(
    weights_path: Path | None = None,
    hand_thresh: float = 0.5,
    first_obj_thresh: float = 0.3,
    second_obj_thresh: float = 0.2,
    device: str | None = None,
):
    """Lazily build and cache a hands23 DefaultPredictor."""
    cache_key = f"{weights_path}_{hand_thresh}_{first_obj_thresh}_{second_obj_thresh}"
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    # Make hands23 imports available
    hands23_str = str(_HANDS23_DIR)
    if hands23_str not in sys.path:
        sys.path.insert(0, hands23_str)

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    # Register custom ROI heads

    wp = weights_path or _ensure_weights()

    cfg = get_cfg()
    cfg.merge_from_file(str(_HANDS23_CONFIG))
    cfg.MODEL.WEIGHTS = str(wp)
    cfg.HAND = hand_thresh
    cfg.FIRSTOBJ = first_obj_thresh
    cfg.SECONDOBJ = second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(hand_thresh, first_obj_thresh, second_obj_thresh)

    if device is not None:
        cfg.MODEL.DEVICE = device
    elif not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    _predictor_cache[cache_key] = predictor
    logger.info("hands23 predictor loaded on %s", cfg.MODEL.DEVICE)
    return predictor


def _run_detection(predictor, pil_image: Image.Image):
    """Run hands23 on a single PIL image and return parsed hand list.

    Uses the deal_output helper from the hands23 demo.
    """
    hands23_str = str(_HANDS23_DIR)
    if hands23_str not in sys.path:
        sys.path.insert(0, hands23_str)
    from demo import deal_output

    img_bgr = np.array(pil_image)[:, :, ::-1].copy()  # RGB -> BGR for cv2/detectron2
    return deal_output(img_bgr, predictor)


def _interaction_score(hand_list) -> float:
    """Compute interaction intensity score for a single frame."""
    score = 0.0
    for hand in hand_list:
        cw = _CONTACT_WEIGHT.get(hand.contactState, 0.0)
        score += hand.pred_score * cw
        if hand.has_first and hand.obj_pred_score is not None:
            score += 0.5 * hand.obj_pred_score
        if hand.has_second and hand.sec_obj_pred_score is not None:
            score += 0.3 * hand.sec_obj_pred_score
    return score


def _extract_object_crops(
    pil_image: Image.Image, hand_list, padding_frac: float = 0.1
) -> list[Image.Image]:
    """Extract padded bbox crops for all first-objects and second-objects."""
    w, h = pil_image.size
    crops: list[Image.Image] = []

    for hand in hand_list:
        for bbox in [hand.obj_bbox, hand.second_obj_bbox]:
            if bbox is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * padding_frac, bh * padding_frac
            cx1 = max(0, int(x1 - px))
            cy1 = max(0, int(y1 - py))
            cx2 = min(w, int(x2 + px))
            cy2 = min(h, int(y2 + py))
            if cx2 > cx1 and cy2 > cy1:
                crops.append(pil_image.crop((cx1, cy1, cx2, cy2)))
    return crops


def _temporal_nms(
    indices_scores: list[tuple[int, float]],
    timestamps: list[float],
    min_gap_s: float = 5.0,
) -> list[int]:
    """Greedy temporal NMS: select indices at least *min_gap_s* apart."""
    sorted_pairs = sorted(indices_scores, key=lambda p: p[1], reverse=True)
    selected: list[int] = []
    selected_ts: list[float] = []

    for idx, _score in sorted_pairs:
        ts = timestamps[idx]
        if all(abs(ts - st) >= min_gap_s for st in selected_ts):
            selected.append(idx)
            selected_ts.append(ts)
    return selected


class HandTool(EvidenceTool):
    """Select frames showing prominent hand-object interactions.

    Uses the hands23 detector (Cheng et al., NeurIPS 2023) to score frames
    by interaction intensity, applies temporal NMS, and optionally re-ranks
    using CLIP similarity between object crops and the question text.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        hand_thresh: float = 0.5,
        first_obj_thresh: float = 0.3,
        second_obj_thresh: float = 0.2,
        clip_model_name: str = "ViT-SO400M-14-SigLIP",
        clip_pretrained: str = "webli",
        device: str | None = None,
        temporal_nms_gap_s: float = 5.0,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._weights_path = Path(weights_path) if weights_path else None
        self._hand_thresh = hand_thresh
        self._first_obj_thresh = first_obj_thresh
        self._second_obj_thresh = second_obj_thresh
        self._clip_model_name = clip_model_name
        self._clip_pretrained = clip_pretrained
        self._temporal_nms_gap_s = temporal_nms_gap_s

        # Lazy-init
        self._predictor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    def _ensure_predictor(self):
        if self._predictor is None:
            wp = self._weights_path or _ensure_weights(self._cache_dir)
            self._predictor = _get_predictor(
                weights_path=wp,
                hand_thresh=self._hand_thresh,
                first_obj_thresh=self._first_obj_thresh,
                second_obj_thresh=self._second_obj_thresh,
                device=self._device,
            )
        return self._predictor

    def _ensure_clip(self):
        if self._clip_model is None:
            import open_clip

            logger.info(
                "Loading CLIP model %s (%s) for hand-tool re-ranking",
                self._clip_model_name,
                self._clip_pretrained,
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._clip_model_name, pretrained=self._clip_pretrained
            )
            self._clip_model = model.to(self._device).eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer(self._clip_model_name)
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    def _clip_score_crops(self, crops: list[Image.Image], question: str) -> float:
        """Return max CLIP cosine similarity between *question* and *crops*."""
        if not crops:
            return 0.0
        model, preprocess, tokenizer = self._ensure_clip()
        images = torch.stack([preprocess(c) for c in crops]).to(self._device)
        tokens = tokenizer([question]).to(self._device)
        with torch.inference_mode():
            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = model.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ txt_feats.T).squeeze(-1)
        return float(sims.max().cpu())

    @property
    def name(self) -> str:
        return "hand"

    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
    ) -> list[Frame]:
        """Select frames with strongest hand-object interactions.

        Args:
            candidate_frames: All available frames for the clip.
            question: VQA question text; used for optional CLIP re-ranking.
            budget: Number of frames to return.

        Returns:
            Selected frames (and their active-object crops) sorted by
            ascending timestamp.
        """
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return list(candidate_frames)

        predictor = self._ensure_predictor()

        # --- Step 1: Score every frame by interaction intensity ---
        frame_scores: list[float] = []
        frame_hands: list[list] = []
        for frame in candidate_frames:
            hands = _run_detection(predictor, frame.image)
            frame_hands.append(hands)
            frame_scores.append(_interaction_score(hands))

        timestamps = [f.timestamp_s for f in candidate_frames]

        # --- Step 2: Temporal NMS ---
        idx_score_pairs = list(enumerate(frame_scores))
        nms_indices = _temporal_nms(idx_score_pairs, timestamps, self._temporal_nms_gap_s)

        # --- Step 3: Select top-K ---
        nms_with_scores = [(i, frame_scores[i]) for i in nms_indices]
        nms_with_scores.sort(key=lambda p: p[1], reverse=True)
        top_indices = [i for i, _ in nms_with_scores[:budget]]

        # --- Step 4: Optional CLIP re-ranking ---
        if question.strip():
            rerank_scores: list[tuple[int, float]] = []
            for i in top_indices:
                crops = _extract_object_crops(candidate_frames[i].image, frame_hands[i])
                clip_sim = self._clip_score_crops(crops, question) if crops else 0.0
                combined = 0.5 * frame_scores[i] + 0.5 * clip_sim
                rerank_scores.append((i, combined))
            rerank_scores.sort(key=lambda p: p[1], reverse=True)
            top_indices = [i for i, _ in rerank_scores[:budget]]

        # --- Step 5: Build output with crops ---
        result: list[Frame] = []
        crop_counter = 0
        for i in sorted(top_indices, key=lambda idx: candidate_frames[idx].timestamp_s):
            result.append(candidate_frames[i])
            crops = _extract_object_crops(candidate_frames[i].image, frame_hands[i])
            for crop_img in crops:
                result.append(
                    Frame(
                        index=-(crop_counter + 1),  # negative index signals a crop
                        timestamp_s=candidate_frames[i].timestamp_s,
                        image=crop_img,
                    )
                )
                crop_counter += 1

        logger.debug(
            "HandTool: selected %d frames + %d crops from %d candidates",
            len(top_indices),
            crop_counter,
            len(candidate_frames),
        )
        return result
