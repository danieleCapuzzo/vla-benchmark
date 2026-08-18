"""
The common contract every VLA policy wrapper must implement.

This module defines the Strategy interface plus the two normalized dataclasses exchanged
between wrappers and the `LiberoEvaluator` (Facade). 
The evaluation loop (Template Method) is written once against this interface 
and is identical for every model, only the concrete implementation varies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class PolicyOutput:
    """
    Normalized output of a single inference call.
    Attributes:
        actions: Array of shape (K, action_dim). K=1 for closed-loop models (OpenVLA, MaIL),
            K>1 for chunked models (OFT, K=8).
        is_chunked: True if the model returns a chunk of actions to be executed open-loop
            before requerying, False if a single action is returned per call.
        raw_latency_s: Wall-clock latency, in seconds, of the model call ONLY (measured by
            the wrapper with `torch.cuda.synchronize()` immediately before/after the call).
            Must exclude any observation preprocessing (resize, normalization, etc.).
    """

    actions: np.ndarray
    is_chunked: bool
    raw_latency_s: float


@dataclass
class PolicyMetadata:
    """
    Static metadata about a loaded model, used for provenance and results tables.
    Attributes:
        model_id: Short identifier ("openvla" | "oft" | "mail").
        variant: Specific checkpoint or config identifying this model instance.
        num_params: Total number of model parameters.
        quant: Quantization mode used ("none" | "8bit" | "4bit").
        num_cameras: Number of camera inputs the model was loaded with.
        action_chunk_size: Nominal chunk-size K (actions-per-real-inference-call) K
        attn_impl: Attention backend in use (must be "sdpa" on this hardware).
        extra: Free-form dict for model-specific details (e.g. unnorm_key).
    """

    model_id: str
    variant: str
    num_params: int
    quant: str
    num_cameras: int
    action_chunk_size: int
    attn_impl: str
    extra: Dict[str, Any] = field(default_factory=dict)


class PolicyWrapper(ABC):
    """
    Common interface that every model wrapper MUST implement (Strategy pattern).

    Adapts a model's native inference API  to this interface (Adapter pattern) 
    so that `LiberoEvaluator` can drive any model through the same rollout loop 
    without knowing model-specific details.
    """

    @abstractmethod
    def load(
        self, quant: str = "none", *, path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Loads model weights, with the requested quantization where applicable.

        Called once per run, before any episodes are executed. Must use the loading
        procedure from the original paper / official harness (see the libero-eval skill
        for per-model quirks), never a generic/simplified path.

        Args:
            quant: Quantization mode, one of "none", "8bit", "4bit". Ignored by models
                with no quantized loading path (MaIL, RoboMamba).
            path: Checkpoint path override. None reuses the checkpoint given in
                `__init__`.
            **kwargs: Model-specific extra loading parameters.

        Raises:
            ValueError: If `quant` is unsupported.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets internal state between episodes.
        Called by the evaluator immediately before starting a new episode (after
        `env.reset()`). For stateless closed-loop models this may be a no-op.
        """

    @abstractmethod
    def predict(self, observation: Dict[str, Any], task_description: str) -> PolicyOutput:
        """
        Runs a single inference call.

        Called by the evaluator once per control step (or once per chunk refill, for
        chunked models). `observation` follows the standard LIBERO key convention
        (`full_image`, optionally `wrist_image`, optionally `state`), already extracted
        from the raw LIBERO env observation but NOT yet resized/normalized for this
        specific model.

        Args:
            observation: Dict with LIBERO-standard keys as produced by the evaluator.
            task_description: Natural-language task instruction for this episode.
        Returns:
            A `PolicyOutput` with the predicted action(s) and the measured model-call
            latency.
        """

    @abstractmethod
    def get_metadata(self) -> PolicyMetadata:
        """
        Returns static metadata describing the currently loaded model.
        Called by the evaluator/CLI after `load()` to populate provenance fields and
        results tables (summary.csv). Must return fully-populated metadata, not a stub.
        """

    @abstractmethod
    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM, in GB, observed so far for this process.
        Called by the evaluator at the end of a run.
        """
