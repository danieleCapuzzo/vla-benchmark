"""
MaIL dataset construction: wraps mail/'s vendored `MultiTaskDataset` (see
`external/mail/dataset/multi_task_dataset_aug.py`) and its language-conditioned variant.
Architecture and checkpoint I/O stay in `core/arch/mail_arch.py`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch

from config.paths import MAIL_DIR
from core.arch.mail_arch import _ensure_mail_importable

# mail/config/benchmark_libero_new.yaml; must exceed every demo's length.
MAX_LEN_DATA = 520

OBS_MODALITIES = {
    "obs": {
        "low_dim": ["ee_ori", "ee_pos", "gripper_states", "joint_states"],
        "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
        "depth": [],
        "scan": [],
    },
    "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
}


def _load_demo_language_embeddings(
    data_dir: List[str], task_suite_name: str, filter_by_attribute: str | None, num_data: int
) -> List[torch.Tensor]:
    """
    One precomputed (1, 512) CLIP embedding per demo index `i`.
    Task name (pkl key) is derived from the filename 
    """
    import pickle

    import h5py
    import numpy as np

    pkl_path = MAIL_DIR / "task_embeddings" / f"{task_suite_name}.pkl"
    with open(pkl_path, "rb") as f:
        embedding_by_task_name: Dict[str, torch.Tensor] = pickle.load(f)

    demo_embeddings: List[torch.Tensor] = []
    for file_path in data_dir:
        task_name = os.path.basename(file_path).split(".")[0][:-5]
        embedding = embedding_by_task_name[task_name]

        with h5py.File(file_path, "r") as f:
            if filter_by_attribute is not None:
                demo_keys_list = [
                    elem.decode("utf-8")
                    for elem in np.array(f[f"mask/{filter_by_attribute}"][:])
                ]
            else:
                demo_keys_list = list(f["data"].keys())
            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])
            num_demos_used = len(indices[:num_data])

        demo_embeddings.extend([embedding] * num_demos_used)

    return demo_embeddings


class LanguageMultiTaskDataset:
    """
    Wraps mail/'s `MultiTaskDataset` to also return each window's precomputed task-language embedding. 
    Composition instead of subclassing.
    """

    def __init__(self, *, task_suite_name: str, **dataset_kwargs: Any) -> None:
        _ensure_mail_importable()
        from dataset.multi_task_dataset_aug import MultiTaskDataset

        self._dataset = MultiTaskDataset(**dataset_kwargs)
        self._demo_embeddings = _load_demo_language_embeddings(
            self._dataset.data_dir,
            task_suite_name=task_suite_name,
            filter_by_attribute=dataset_kwargs.get("filter_by_attribute"),
            num_data=dataset_kwargs.get("num_data", 10),
        )
        assert len(self._demo_embeddings) == self._dataset.num_data

    def get_all_actions(self) -> torch.Tensor:
        return self._dataset.get_all_actions()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        agentview_rgb, eye_in_hand_rgb, act, mask = self._dataset[idx]
        i, _start, _end = self._dataset.slices[idx]
        return agentview_rgb, eye_in_hand_rgb, act, mask, self._demo_embeddings[i]


def build_mail_dataset(
    *, suite: str, task_suite_name: str, data_dir: str,
    arch: Any, num_demos: int | None, language: bool, 
):
    """
    Builds the MaIL training dataset for one suite.
    Args:
        suite: Short suite name (unused directly; kept for caller-side logging symmetry).
        task_suite_name: LIBERO benchmark suite dir name (e.g. "libero_10"), used to look up
            precomputed language embeddings when `language=True`.
        data_dir: Directory containing the suite's `*_demo.hdf5` files.
        arch: `MailArchConfig` for this suite (supplies `action_dim`, `window_size`).
        num_demos: Demos per task, or None for all 50.
        language: Whether to attach precomputed CLIP task embeddings per window.
    Returns:
        A `MultiTaskDataset` or `LanguageMultiTaskDataset` instance.
    """
    _ensure_mail_importable()
    from dataset.multi_task_dataset_aug import MultiTaskDataset

    dataset_kwargs = dict(
        data_directory=str(data_dir),
        obs_keys="rgb",
        obs_modalities=OBS_MODALITIES,
        dataset_keys=["actions", "dones", "obs", "rewards", "states"],
        filter_by_attribute=None,
        padding=False,
        device="cpu",
        obs_dim=64,
        action_dim=arch.action_dim,
        state_dim=110,
        max_len_data=MAX_LEN_DATA,
        window_size=arch.window_size,
        num_data=num_demos if num_demos is not None else 50,
        data_aug=False,
    )
    if language:
        return LanguageMultiTaskDataset(task_suite_name=task_suite_name, **dataset_kwargs)
    return MultiTaskDataset(**dataset_kwargs)
