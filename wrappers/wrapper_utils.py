"""
Shared boilerplate for the OpenVLA-family wrappers (base OpenVLA and OpenVLA-OFT).
Contains minimal reimplementations for two functions for newer Blackwell hardware.

Why this file:
    OFT's own `get_vla`/`get_vla_action` (openvla-oft/experiments/robot/openvla_utils.py) are
    already Blackwell-safe, never hardcode `flash_attention_2` and are imported as-is by
    `wrappers/oft_wrapper.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from config.constants import OPENVLA_INPUT_RESOLUTION  # noqa: F401 (re-exported)

# attention backend forced on Blackwell/sm_120 (flash-attn is not usable, see CLAUDE.md).
ATTN_IMPL = "sdpa"

# official OpenVLA v0.1 system prompt, prepended only for that checkpoint variant.
_SYSTEM_PROMPT_V01 = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_openvla_prompt(task_description: str, base_vla_name: str = "") -> str:
    """Builds the OpenVLA prompt string, verbatim format from the official harness
    (openvla/experiments/robot/openvla_utils.py::get_vla_action)."""
    if "openvla-v01" in base_vla_name:
        return (
            f"{_SYSTEM_PROMPT_V01} USER: What action should the robot take to "
            f"{task_description.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


def register_openvla_auto_classes() -> None:
    """Registers the OpenVLA model/config/processor with HF's Auto classes.

    Copied from openvla/experiments/robot/openvla_utils.py::get_vla. Required so
    `AutoModelForVision2Seq`/`AutoProcessor.from_pretrained` resolve the "openvla" model
    type without a local `--trust-remote-code` entry point. Safe to call more than once
    (`already registered` raises ValueError from HF internals, guarded here).
    """
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        # already registered (e.g. a previous suite's load() in the same process).
        pass


def load_openvla_model(checkpoint: str, quant: str) -> Any:
    """Loads a base-OpenVLA-family checkpoint with the correct attn backend + dtype for
    this hardware, branching on quantization exactly like `old_scripts/quant_inference.py`
    (validated standalone on the RTX 5080 prior to wrapper integration).

    Args:
        checkpoint: HF Hub id or local path of the checkpoint.
        quant: one of "none", "8bit", "4bit".

    Returns:
        The loaded `OpenVLAForActionPrediction` model, in eval mode, placed on the correct
        device (`.to("cuda")` for unquantized, `device_map="auto"` for quantized).
    """
    from transformers import AutoModelForVision2Seq

    register_openvla_auto_classes()

    if quant == "none":
        vla = (
            AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=ATTN_IMPL,
            )
            .to("cuda")
            .eval()
        )
    else:
        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            quantization_config=quant_config,
            device_map="auto",  # required for quantized models; never call .to() after
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
        ).eval()

    return vla


def resize_openvla_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resizes `image` to `size` x `size` via the official OpenVLA/Octo dataloader scheme:
    JPEG encode/decode round-trip then lanczos3 resize. Verbatim port of
    `openvla/experiments/robot/libero/libero_utils.py::resize_image` (paper-critical --
    plain bilinear/nearest upscaling shifts the input distribution away from training).
    """
    import tensorflow as tf

    img = tf.image.encode_jpeg(image)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (size, size), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def center_crop_and_resize(
    image: Image.Image, crop_scale: float, size: int
) -> Image.Image:
    """
    Center-crops `image` to `crop_scale` of its original area, then resizes back to
    `size` x `size`.

    Mirrors `openvla/experiments/robot/openvla_utils.py::crop_and_resize` without tensorflow dependencies.
    Using this function is required in OpenVLA to mirror how it is trained, skipping it causes a train/inference
    input distribution mismatch tanking success rates. The official LIBERO eval harness asserts `center_crop=True`
    for this reason.

    """
    width, height = image.size
    # official implementation crops on AREA (crop_scale) but the linear per-axis factor is
    # sqrt(crop_scale), not crop_scale itself.
    linear_scale = crop_scale**0.5
    crop_width = width * linear_scale
    crop_height = height * linear_scale
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((size, size))


def prepare_openvla_inputs(
    processor: Any, image: np.ndarray, task_description: str, quant: str
) -> dict:
    """
    Builds the processor inputs for a single OpenVLA-family inference call.

    `image` must already be rotated 180 degrees (see `core/core_utils.py::get_libero_image`)
    but NOT yet resized -- this function resizes it to `OPENVLA_INPUT_RESOLUTION` via the
    official lanczos3+JPEG round-trip (`resize_openvla_image`), applies the 0.9-area
    center-crop the LIBERO-finetuned checkpoints were trained with, builds the official
    prompt, and casts `pixel_values` to bfloat16 (unquantized) or float16 (8bit/4bit).
    """
    image = resize_openvla_image(image, OPENVLA_INPUT_RESOLUTION)
    pil_image = Image.fromarray(image).convert("RGB")
    pil_image = center_crop_and_resize(
        pil_image, crop_scale=0.9, size=OPENVLA_INPUT_RESOLUTION
    )
    prompt = build_openvla_prompt(task_description)

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to("cuda")
    pixel_dtype = torch.bfloat16 if quant == "none" else torch.float16
    inputs["pixel_values"] = inputs["pixel_values"].to(pixel_dtype)
    return inputs
