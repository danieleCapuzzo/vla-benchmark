"""
Single source of truth for filesystem layout: repo root, the `external/` vendored repos, and
where trained checkpoints live. Import from here instead of re-deriving
`Path(__file__).resolve().parent.parent / "<name>"` at each call site.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = REPO_ROOT / "external"

LIBERO_DIR = EXTERNAL / "LIBERO"
OPENVLA_DIR = EXTERNAL / "openvla"
OPENVLA_OFT_DIR = EXTERNAL / "openvla-oft"
MAIL_DIR = EXTERNAL / "mail"
ROBOMAMBA_DIR = EXTERNAL / "robo_mamba"
ROBOMAMBA_SRC = ROBOMAMBA_DIR / "src"

LIBERO_DATASETS = LIBERO_DIR / "libero" / "datasets"
TRAINED_DIR = REPO_ROOT / "trained"
