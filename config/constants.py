"""
Constants shared across run_eval.py, core/evaluator.py, and the wrappers. Centralized so a
per-model value (camera count, render resolution) has exactly one definition instead of a
wrapper-local copy that can drift from a CLI-facing dict.
"""

from __future__ import annotations

ALL_SUITES = ["spatial", "object", "goal", "long"]

SUITE_TO_DATASET_DIR = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}

# LIBERO episode step budgets, keyed by task_suite_name (see core/evaluator.py).
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Cameras each model consumes. Wrappers import NUM_CAMERAS from here so their internal
# constant and this dict cannot drift apart.
NUM_CAMERAS_BY_MODEL = {
    "openvla": 1,
    "oft": 2,
    "mail": 2,
    "robomamba": 1,
}

# Render resolution each model was trained on. OpenVLA/OFT/RoboMamba train on 256px re-rendered
# frames (datasets/libero_*_regen); MaIL was trained on the native-128 LIBERO hdf5 demos and
# should be evaluated at the same resolution (upstream: mail/simulation/benchmark_sim.py
# camera_heights/widths=128).
DEFAULT_RENDER_RESOLUTION = 256
RENDER_RESOLUTION_BY_MODEL = {
    "openvla": 256,
    "oft": 256,
    "mail": 128,
    "robomamba": 256,
}

# Final input resolution to the OpenVLA-family vision backbones (both openvla and oft resize
# their 256px render down to this before the model sees it).
OPENVLA_INPUT_RESOLUTION = 224

# MaIL's native training/eval resolution (see mail_wrapper.py and core/mail_arch.py).
MAIL_IMAGE_SIZE = 128

# RoboMamba's CLIP ViT-L/14@224 backbone, fed from the 256px regenerated renders.
ROBOMAMBA_IMAGE_SIZE = OPENVLA_INPUT_RESOLUTION
