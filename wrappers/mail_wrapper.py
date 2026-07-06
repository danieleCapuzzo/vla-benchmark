"""
Adapter for MambaImitationLearning MaIL (see https://github.com/ALRhub/MaIL).

# TODO: implement is according to doc/ARCHITECTURE (PloicyWrapper contract)
# remember to use a new custom environment for Mamba
"""

from __future__ import annotations

from typing import Any, Dict

from core.factory import PolicyFactory
from core.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper


@PolicyFactory.register("mail")
class MailWrapper(PolicyWrapper):
    """Strategy/Adapter skeleton for MaIL. Not implemented yet."""

    def __init__(self, checkpoint: str, unnorm_key: str) -> None:
        self._checkpoint = checkpoint
        self._unnorm_key = unnorm_key

    def load(self, quant: str = "none", num_cameras: int = 1, **kwargs: Any) -> None:
        raise NotImplementedError("MailWrapper.load is not implemented yet")

    def reset(self) -> None:
        raise NotImplementedError("MailWrapper.reset is not implemented yet")

    def predict(
        self, observation: Dict[str, Any], task_description: str
    ) -> PolicyOutput:
        raise NotImplementedError("MailWrapper.predict is not implemented yet")

    def get_metadata(self) -> PolicyMetadata:
        raise NotImplementedError("MailWrapper.get_metadata is not implemented yet")

    def get_vram_footprint_gb(self) -> float:
        raise NotImplementedError(
            "MailWrapper.get_vram_footprint_gb is not implemented yet"
        )
