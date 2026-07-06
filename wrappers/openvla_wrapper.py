"""
Adapter for OpenVLA (see https://github.com/openvla/openvla).

# TODO: implement is according to doc/ARCHITECTURE (PloicyWrapper contract)
"""

from __future__ import annotations

from typing import Any, Dict

from core.factory import PolicyFactory
from core.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper


@PolicyFactory.register("openvla")
class OpenVlaWrapper(PolicyWrapper):
    """Strategy/Adapter skeleton for base OpenVLA. Not implemented yet."""

    def __init__(self, checkpoint: str, unnorm_key: str) -> None:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        self._checkpoint = checkpoint
        self._unnorm_key = unnorm_key

    def load(self, quant: str = "none", num_cameras: int = 1, **kwargs: Any) -> None:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        # If the official openvla/ repo assumes cwd == openvla/ (mirroring OftWrapper's
        # os.chdir(_OFT_REPO_ROOT) pattern in wrappers/oft_wrapper.py), redirect cwd to this
        # model's OWN repo folder here -- do not reuse OFT's path.
        raise NotImplementedError("OpenVlaWrapper.load is not implemented yet")

    def reset(self) -> None:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        raise NotImplementedError("OpenVlaWrapper.reset is not implemented yet")

    def predict(
        self, observation: Dict[str, Any], task_description: str
    ) -> PolicyOutput:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        raise NotImplementedError("OpenVlaWrapper.predict is not implemented yet")

    def get_metadata(self) -> PolicyMetadata:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        raise NotImplementedError("OpenVlaWrapper.get_metadata is not implemented yet")

    def get_vram_footprint_gb(self) -> float:
        # TODO: see doc/ARCHITECTURE.md §4.3 / libero-eval skill "OpenVLA base".
        raise NotImplementedError(
            "OpenVlaWrapper.get_vram_footprint_gb is not implemented yet"
        )
