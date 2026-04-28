"""Qwen3-VL inference wrapper for egocentric video QA."""

from __future__ import annotations

import logging
import math
import re
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from eer.data.frames import Frame

logger = logging.getLogger(__name__)

_ANSWER_LETTERS = list("ABCDE")


@dataclass
class VLMResult:
    """Output from a single VQA inference call."""

    predicted_letter: str  # A–E (or "?" on parse failure)
    log_probs: dict[str, float]  # log-prob for each letter A–E
    raw_output: str  # raw decoded text from the model


def _build_choice_prompt(question: str, choices: list[str]) -> str:
    """Format question + choices into a multiple-choice prompt string."""
    lines = [question]
    for letter, text in zip(_ANSWER_LETTERS, choices):
        lines.append(f"{letter}. {text}")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def _extract_letter(text: str) -> str:
    """Pull the first A–E letter out of a raw model response."""
    text = text.strip()
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    # Fallback: first character if it happens to be a letter
    if text and text[0].upper() in _ANSWER_LETTERS:
        return text[0].upper()
    return "?"


def _extract_log_probs(
    scores: tuple[torch.Tensor, ...],
    tokenizer: Any,
) -> dict[str, float]:
    """Extract per-letter log-probabilities from the first generated token.

    Args:
        scores: Tuple of logit tensors from ``model.generate``,
                each of shape ``(batch, vocab_size)``.
        tokenizer: The tokenizer (used to get token ids for A–E).

    Returns:
        Dict mapping letter → log-prob float.
    """
    if not scores:
        return {l: float("-inf") for l in _ANSWER_LETTERS}

    first_logits = scores[0][0]  # (vocab_size,)
    log_probs = torch.log_softmax(first_logits.float(), dim=-1)

    result: dict[str, float] = {}
    for letter in _ANSWER_LETTERS:
        # encode without special tokens; take the last token id
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            result[letter] = log_probs[ids[-1]].item()
        else:
            result[letter] = float("-inf")
    return result


def _clear_sampling_defaults(generation_config: Any) -> None:
    """Remove sampling-only defaults so deterministic generation is quiet."""
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, attr):
            setattr(generation_config, attr, None)


def _format_timestamp(timestamp_s: float) -> str:
    """Format a frame timestamp compactly for the prompt."""
    minutes = int(timestamp_s // 60)
    seconds = timestamp_s - minutes * 60
    if minutes:
        return f"{minutes:d}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def _append_auxiliary_frame_content(
    content: list[dict],
    auxiliary_frames: list[Image.Image | Frame] | None,
) -> None:
    """Append auxiliary images, adding timestamp labels when Frame metadata exists."""
    if not auxiliary_frames:
        return

    for i, item in enumerate(auxiliary_frames, start=1):
        if isinstance(item, Frame):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Auxiliary evidence image {i} "
                        f"(timestamp {_format_timestamp(item.timestamp_s)}):"
                    ),
                }
            )
            content.append({"type": "image", "image": item.image})
        else:
            content.append({"type": "image", "image": item})


def _prepare_vision_inputs(
    messages: list[dict],
    image_patch_size: int = 16,
) -> tuple[Any, Any, dict[str, Any]]:
    """Process Qwen vision messages with video metadata when supported."""
    from qwen_vl_utils import process_vision_info

    signature = inspect.signature(process_vision_info)
    kwargs: dict[str, Any] = {"image_patch_size": image_patch_size}
    if "return_video_kwargs" in signature.parameters:
        kwargs["return_video_kwargs"] = True
    if "return_video_metadata" in signature.parameters:
        kwargs["return_video_metadata"] = True

    processed = process_vision_info(messages, **kwargs)
    if len(processed) == 2:
        image_inputs, video_inputs = processed
        return image_inputs, video_inputs, {}

    image_inputs, video_inputs, video_kwargs = processed
    processor_kwargs = dict(video_kwargs or {})

    if video_inputs is not None and "return_video_metadata" in kwargs:
        video_inputs, video_metadata = zip(*video_inputs)
        processor_kwargs["video_metadata"] = list(video_metadata)
        processor_kwargs.setdefault("do_resize", False)
        video_inputs = list(video_inputs)

    return image_inputs, video_inputs, processor_kwargs


class QwenVLM:
    """Thin wrapper around Qwen3-VL for multiple-choice VQA.

    The model and processor are loaded once at construction and reused
    across all inference calls.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        """Load Qwen3-VL model and processor.

        Args:
            model_name: HuggingFace model id or local path.
            dtype: torch dtype string ("bfloat16", "float16", "float32").
            device_map: Passed directly to ``from_pretrained``.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        logger.info("Loading Qwen3-VL model: %s (dtype=%s)", model_name, dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()
        _clear_sampling_defaults(self.model.generation_config)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Model loaded.")

    def answer_open_ended(
        self,
        video_path: str | Path | None,
        auxiliary_frames: list[Image.Image | Frame] | None,
        question: str,
        video_fps: float = 1.0,
    ) -> str:
        """Run open-ended VQA inference.

        Args:
            video_path: Path to the video clip file, or None.
            auxiliary_frames: List of PIL images or Frame objects selected by an
                evidence tool, or None. Frame objects add timestamp labels to
                the prompt.
            question: The question text.
            video_fps: FPS to sample the video at if passing the full video.

        Returns:
            The raw decoded text string answer.
        """
        content: list[dict] = []

        if video_path is not None:
            logger.info("Passing video to Qwen with requested sample fps=%.3f", video_fps)
            content.append({"type": "video", "video": str(video_path), "fps": video_fps})

        _append_auxiliary_frame_content(content, auxiliary_frames)

        content.append({"type": "text", "text": question + " Please answer concisely."})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, processor_kwargs = _prepare_vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug("Open-ended VQA output: %r", raw_output)
        return raw_output

    def answer_multiple_choice(
        self,
        video_path: str | Path | None,
        auxiliary_frames: list[Image.Image | Frame] | None,
        question: str,
        choices: list[str],
        video_fps: float = 1.0,
    ) -> VLMResult:
        """Run multiple-choice VQA inference.

        Supports three modes:
        - **video only**: pass *video_path*, leave *auxiliary_frames* None.
        - **video + aux frames**: pass both; aux frames are appended as images.
        - **aux frames only**: pass *auxiliary_frames*, leave *video_path* None.

        Args:
            video_path: Path to the video clip file, or None.
            auxiliary_frames: List of PIL images or Frame objects selected by an
                evidence tool, or None. Frame objects add timestamp labels to
                the prompt.
            question: The question text.
            choices: List of exactly 5 answer strings (indexed A–E).
            video_fps: FPS to sample the video at if passing the full video.

        Returns:
            VLMResult with the predicted letter, per-letter log-probs, and
            the raw decoded output string.
        """
        content: list[dict] = []

        if video_path is not None:
            logger.info("Passing video to Qwen with requested sample fps=%.3f", video_fps)
            content.append({"type": "video", "video": str(video_path), "fps": video_fps})

        _append_auxiliary_frame_content(content, auxiliary_frames)

        prompt = _build_choice_prompt(question, choices)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, processor_kwargs = _prepare_vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )

        # Move inputs to the model's device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # Decode the generated tokens (everything after the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=True)

        predicted = _extract_letter(raw_output)
        log_probs = _extract_log_probs(outputs.scores, self.processor.tokenizer)

        logger.debug("VQA output: %r → letter=%s", raw_output, predicted)
        return VLMResult(
            predicted_letter=predicted,
            log_probs=log_probs,
            raw_output=raw_output,
        )
