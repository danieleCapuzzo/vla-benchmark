"""Instantiates the correct PolicyWrapper for a given model_id (Factory pattern).

Adding a 4th model requires no changes to this file: a new wrapper module registers
itself via the `PolicyFactory.register` decorator (Open/Closed Principle), see
doc/ARCHITECTURE.md §7.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from wrappers.policy_wrapper import PolicyWrapper


class PolicyFactory:
    """Registry mapping model_id -> PolicyWrapper subclass, and factory method."""

    _registry: Dict[str, Type[PolicyWrapper]] = {}

    @classmethod
    def register(cls, model_id: str) -> Callable[[Type[PolicyWrapper]], Type[PolicyWrapper]]:
        """Class decorator that registers a PolicyWrapper subclass under `model_id`.

        Usage:
            @PolicyFactory.register("oft")
            class OftWrapper(PolicyWrapper):
                ...

        Args:
            model_id: Short identifier used on the CLI (e.g. "openvla", "oft", "mail").

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def _decorator(wrapper_cls: Type[PolicyWrapper]) -> Type[PolicyWrapper]:
            cls._registry[model_id] = wrapper_cls
            return wrapper_cls

        return _decorator

    @classmethod
    def create(cls, model_id: str, **kwargs: Any) -> PolicyWrapper:
        """Instantiates the wrapper registered for `model_id`.

        Args:
            model_id: Identifier of the model to instantiate (must be registered).
            **kwargs: Forwarded to the wrapper's constructor.

        Returns:
            A new instance of the registered `PolicyWrapper` subclass.

        Raises:
            ValueError: If `model_id` is not registered. The error message lists all
                currently valid model_ids.
        """
        if model_id not in cls._registry:
            valid = ", ".join(sorted(cls._registry)) or "<none registered>"
            raise ValueError(f"Unknown model_id '{model_id}'. Valid model_ids: {valid}")
        return cls._registry[model_id](**kwargs)
