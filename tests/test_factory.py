"""Unit tests for core/factory.py: registration, creation, unknown-id error."""

import pytest

from core.factory import PolicyFactory
from wrappers.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper


class _DummyWrapper(PolicyWrapper):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self, quant="none", *, path=None, trunk_path=None, head_path=None, **kwargs):
        pass

    def reset(self):
        pass

    def predict(self, observation, task_description):
        return PolicyOutput(actions=None, is_chunked=False, raw_latency_s=0.0)

    def get_metadata(self):
        return PolicyMetadata(
            model_id="dummy",
            variant="v0",
            num_params=0,
            quant="none",
            num_cameras=1,
            action_chunk_size=1,
            attn_impl="sdpa",
            extra={},
        )

    def get_vram_footprint_gb(self):
        return 0.0


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolates the class-level registry so tests don't leak state into each other."""
    original = dict(PolicyFactory._registry)
    yield
    PolicyFactory._registry = original


def test_register_and_create():
    PolicyFactory.register("dummy")(_DummyWrapper)
    wrapper = PolicyFactory.create("dummy", foo="bar")
    assert isinstance(wrapper, _DummyWrapper)
    assert wrapper.kwargs == {"foo": "bar"}


def test_register_is_decorator_returning_class_unchanged():
    decorated = PolicyFactory.register("dummy2")(_DummyWrapper)
    assert decorated is _DummyWrapper


def test_create_unknown_model_id_raises_with_valid_ids_listed():
    PolicyFactory.register("dummy")(_DummyWrapper)
    PolicyFactory.register("dummy2")(_DummyWrapper)
    with pytest.raises(ValueError) as exc_info:
        PolicyFactory.create("does_not_exist")
    message = str(exc_info.value)
    assert "does_not_exist" in message
    assert "dummy" in message
    assert "dummy2" in message
