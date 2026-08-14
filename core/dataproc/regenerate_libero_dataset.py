#!/usr/bin/env python
"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the live LIBERO
environment, instead of reusing the shipped hdf5's stored frames.

Ported from external/openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py
(the script OFT/OpenVLA use to build their own LIBERO training data), adapted to:
  - run from the repo root instead of assuming cwd == openvla-oft/
  - use core.core_utils.get_libero_env / get_libero_dummy_action instead of the vendored
    experiments.robot.libero.libero_utils copies
  - take --metainfo-path as a CLI arg instead of a hardcoded relative path
  - --overwrite instead of an interactive y/n prompt, for unattended runs
  - --num-workers to regenerate tasks in parallel (one process + renderer per task),
    since upstream's per-task loop is single-process and CPU/render bound

Notes (from upstream):
    - Image observations are saved at IMAGE_RESOLUTION (default 256px, not the shipped
      hdf5's 128px) -- this is what removes the render-time train/eval mismatch (see
      doc/ROBOMAMBA.md): a head trained on these frames sees the same renderer at eval.
    - Transitions with "no-op" (near-zero) actions are filtered out (see `is_noop`).
    - Unsuccessful demonstrations are filtered out (only `done` episodes are kept).
    - Orientation is left as the live env returns it -- unlike upstream's separate hdf5->RLDS
      conversion step (not ported here), no 180-degree rotation is applied. Downstream
      consumers rotate at read time if their training data expects it (see
      core.core_utils.get_libero_image / to_upright).

Output schema is key-compatible with the original hdf5s (`obs/agentview_rgb`,
`obs/eye_in_hand_rgb`, `actions`, `states`, `obs/ee_pos`, `obs/ee_ori`, `obs/joint_states`,
`obs/gripper_states`), so it drops into any existing `--data-dir`.

Relative `--libero_raw_data_dir` / `--libero_target_dir` / `--metainfo-path` values resolve
against the repo root, not the caller's cwd, so this runs the same from anywhere:
    python core/dataproc/regenerate_libero_dataset.py \\
        --libero_task_suite libero_spatial \\
        --libero_raw_data_dir datasets/libero_spatial \\
        --libero_target_dir datasets/libero_spatial_regen
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from tqdm.auto import tqdm
from libero.libero import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import REPO_ROOT  # noqa: E402
from core.core_utils import get_libero_dummy_action, get_libero_env  # noqa: E402


def _resolve(path: str) -> str:
    """Repo-relative CLI paths -> absolute, independent of the caller's cwd."""
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)

IMAGE_RESOLUTION = 256
MODEL_FAMILY = "openvla"  # only picks the dummy-action shape in core_utils; harmless label


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def process_task(
    task_id: int,
    libero_task_suite: str,
    libero_raw_data_dir: str,
    libero_target_dir: str,
    resolution: int,
    progress_queue: "mp.Queue",
) -> tuple[str, dict, int, int, int]:
    """
    Regenerates the hdf5 for a single task. Runs in its own process and returns plain data.

    Per-demo progress is pushed onto `progress_queue` instead of drawn locally: separate
    processes each running their own tqdm bar have no shared registry of cursor position, so
    concurrent bars corrupt each other's terminal lines. A single bar owned by the main
    process (fed by this queue) avoids that entirely.
    """
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[libero_task_suite]()
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, MODEL_FAMILY, resolution=resolution)

    num_replays = 0
    num_success = 0
    num_noops = 0
    task_metainfo: dict = {}

    orig_data_path = os.path.join(libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    num_demos = len(orig_data.keys())

    new_data_path = os.path.join(libero_target_dir, f"{task.name}_demo.hdf5")
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    for i in range(num_demos):
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        env.reset()
        env.set_init_state(orig_states[0])
        for _ in range(10):
            obs, reward, done, info = env.step(get_libero_dummy_action(MODEL_FAMILY))

        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []

        for _, action in enumerate(orig_actions):
            prev_action = actions[-1] if len(actions) > 0 else None
            if is_noop(action, prev_action):
                num_noops += 1
                continue

            if states == []:
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                states.append(env.sim.get_state().flatten())
                robot_states.append(
                    np.concatenate(
                        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                    )
                )

            actions.append(action)

            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(
                np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
            )
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

            obs, reward, done, info = env.step(action.tolist())

        if done:
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            assert len(actions) == len(agentview_images)

            ep_data_grp = grp.create_group(f"demo_{i}")
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=np.stack(states))
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)

            num_success += 1

        num_replays += 1

        task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{i}"
        task_metainfo.setdefault(task_key, {})[episode_key] = {
            "success": bool(done),
            "initial_state": orig_states[0].tolist(),
        }

        progress_queue.put(1)

    orig_data_file.close()
    new_data_file.close()
    env.close()
    return task_description, task_metainfo, num_replays, num_success, num_noops


def _process_task_star(worker_args: tuple) -> tuple[str, dict, int, int, int]:
    """`imap_unordered` only passes a single argument per call; unpack the tuple here."""
    return process_task(*worker_args)


def main(args: argparse.Namespace) -> None:
    args.libero_raw_data_dir = _resolve(args.libero_raw_data_dir)
    args.libero_target_dir = _resolve(args.libero_target_dir)
    if args.metainfo_path:
        args.metainfo_path = _resolve(args.metainfo_path)

    target_dir = Path(args.libero_target_dir)
    if target_dir.is_dir() and not args.overwrite:
        raise FileExistsError(
            f"Target directory already exists at {target_dir}. Pass --overwrite to replace it."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    metainfo_path = (
        Path(args.metainfo_path)
        if args.metainfo_path
        else target_dir / f"{args.libero_task_suite}_metainfo.json"
    )

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    num_workers = args.num_workers or num_tasks_in_suite

    print("=== REGENERATE LIBERO DATASET ===")
    print(f"= source_dir: {args.libero_raw_data_dir}")
    print(f"= dest_dir: {args.libero_target_dir}")
    print(f"= renderer setting {args.resolution}x{args.resolution}")
    print(f"= {num_tasks_in_suite} tasks / {num_workers} workers\n")

    total_demos = 0
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        raw_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        with h5py.File(raw_path, "r") as f:
            total_demos += len(f["data"].keys())

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()

    worker_args = [
        (
            task_id,
            args.libero_task_suite,
            args.libero_raw_data_dir,
            args.libero_target_dir,
            args.resolution,
            progress_queue,
        )
        for task_id in range(num_tasks_in_suite)
    ]

    metainfo_json_dict: dict = {}
    num_replays = 0
    num_success = 0
    num_noops = 0

    # Each task runs in its own process (own MuJoCo/robosuite renderer). Separate processes
    # have no shared tqdm registry, so per-process bars can't coordinate cursor position and
    # corrupt each other's terminal lines. Instead, every process pushes progress ticks onto
    # `progress_queue`; only this single bar in the main process draws to the terminal.
    overall_bar = tqdm(total=total_demos, desc="demos done", unit="demo")
    pool = ctx.Pool(num_workers)
    try:
        async_result = pool.map_async(_process_task_star, worker_args)
        pool.close()
        completed = 0
        while completed < total_demos:
            progress_queue.get()
            completed += 1
            overall_bar.update(1)
        results = async_result.get()
    finally:
        overall_bar.close()
        pool.join()

    for task_description, task_metainfo, task_replays, task_success, task_noops in results:
        metainfo_json_dict.update(task_metainfo)
        num_replays += task_replays
        num_success += task_success
        num_noops += task_noops

    with open(metainfo_path, "w") as f:
        json.dump(metainfo_json_dict, f, indent=2)

    print(
        f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} "
        f"({num_success / num_replays * 100:.1f} %)  no-ops filtered: {num_noops}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        required=True,
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        required=True,
        help="Path to directory containing the raw hdf5 dataset, e.g. "
        "datasets/libero_spatial",
    )
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        required=True,
        help="Path to write the regenerated dataset, e.g. "
        "datasets/libero_spatial_256",
    )
    parser.add_argument(
        "--resolution", type=int, default=IMAGE_RESOLUTION, help="Square render resolution."
    )
    parser.add_argument(
        "--metainfo-path",
        type=str,
        default=None,
        help="Default: <libero_target_dir>/<libero_task_suite>_metainfo.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite libero_target_dir if it already exists (default: error out).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of tasks to regenerate in parallel, each in its own process with its own "
        "renderer (default: one worker per task in the suite).",
    )
    main(parser.parse_args())
