"""LiberoEvaluator: Facade over LIBERO environment setup + rollout loop (Template Method).

Hides the complexity of constructing a LIBERO environment for a given suite/task, running
episodes against any PolicyWrapper, and collecting success/latency metrics. Depends ONLY on
the PolicyWrapper interface -- never on model-specific calls -- so the identical loop drives
OpenVLA, OFT, or MaIL (Template Method: only wrapper.predict()'s internals vary).

Reuses the official OFT/LIBERO preprocessing helpers rather than reimplementing them
(doc/ARCHITECTURE.md §4.1): observation extraction/rotation and the gripper action
convention are taken verbatim from openvla-oft/experiments/robot/{libero/libero_utils,
robot_utils}.py, mirroring the split of the official `prepare_observation`/`process_action`
functions at their natural seam (env-level concerns here; model-specific resize lives in
the wrapper, see wrappers/oft_wrapper.py).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_OFT_REPO_ROOT = Path(__file__).resolve().parent.parent / "openvla-oft"
if str(_OFT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFT_REPO_ROOT))

from core.policy_wrapper import PolicyWrapper  # noqa: E402


# LIBERO-generic per-suite step budgets
TASK_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoEvaluator:
    """Facade: runs a full suite evaluation for a given PolicyWrapper."""

    def __init__(self, task_suite_name: str, num_steps_wait: int = 10) -> None:
        """Creates the evaluator for one LIBERO task suite.

        Args:
            task_suite_name: LIBERO benchmark suite id (e.g. "libero_spatial").
            num_steps_wait: Number of no-op steps at episode start to let objects settle,
                matching the official harness default.
        """
        self._task_suite_name = task_suite_name
        self._num_steps_wait = num_steps_wait
        self._max_steps = TASK_MAX_STEPS[task_suite_name]

    def run(
        self, wrapper: PolicyWrapper, num_rollouts_per_task: int
    ) -> List[Dict[str, Any]]:
        """Runs `num_rollouts_per_task` episodes for every task in the suite.

        Args:
            wrapper: A loaded PolicyWrapper (load() must already have been called).
            num_rollouts_per_task: Number of episodes to run per task (frozen protocol
                value is 20; see doc/ARCHITECTURE.md §2).

        Returns:
            A list of per-episode result dicts, one per (task, trial), each with keys
            "suite", "task_id", "trial", "success", "num_steps", "latencies_s", "lat_mean_s",
            "is_chunked", "chunk_size".
        """
        from libero.libero import benchmark

        from experiments.robot.libero.libero_utils import get_libero_env

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self._task_suite_name]()

        print(
            f"[{self._task_suite_name}] {task_suite.n_tasks} tasks x "
            f"{num_rollouts_per_task} rollouts = {task_suite.n_tasks * num_rollouts_per_task} episodes"
        )

        episodes: List[Dict[str, Any]] = []
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(
                task, model_family="openvla", resolution=256
            )

            print(
                f"[{self._task_suite_name}] task {task_id + 1}/{task_suite.n_tasks}: "
                f'"{task_description}"'
            )
            successes = 0
            try:
                for trial in range(num_rollouts_per_task):
                    initial_state = initial_states[trial]
                    episode = self._run_episode(
                        wrapper=wrapper,
                        env=env,
                        task_description=task_description,
                        initial_state=initial_state,
                        task_id=task_id,
                        trial=trial,
                    )
                    episodes.append(episode)
                    successes += int(episode["success"])
                    print(
                        f"  trial {trial + 1}/{num_rollouts_per_task}  "
                        f"{'OK  ' if episode['success'] else 'FAIL'}  "
                        f"steps={episode['num_steps']:<4d} "
                        f"lat={episode['lat_mean_s'] * 1000:.0f}ms  "
                        f"(success {successes}/{trial + 1})"
                    )
            finally:
                # explicitly release MuJoco sim + EGL renderer
                env.close()

        return episodes

    def _run_episode(
        self,
        wrapper: PolicyWrapper,
        env: Any,
        task_description: str,
        initial_state: np.ndarray,
        task_id: int,
        trial: int,
    ) -> Dict[str, Any]:
        """Runs a single episode and returns its result dict."""
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        env.reset()
        obs = env.set_init_state(initial_state)
        wrapper.reset()

        action_queue: deque = deque()
        latencies_s: List[float] = []
        success = False
        num_steps = 0
        is_chunked = True
        chunk_size = 0

        t = 0
        while t < self._max_steps + self._num_steps_wait:
            if t < self._num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            if len(action_queue) == 0:
                observation = self._build_observation(obs)
                output = wrapper.predict(observation, task_description)
                latencies_s.append(output.raw_latency_s)
                is_chunked = output.is_chunked
                chunk_size = int(output.actions.shape[0])
                action_queue.extend(output.actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, _, done, _ = env.step(action.tolist())
            num_steps += 1
            if done:
                success = True
                break
            t += 1

        lat_mean_s = float(np.mean(latencies_s)) if latencies_s else 0.0
        return {
            "suite": self._task_suite_name,
            "task_id": task_id,
            "trial": trial,
            "success": success,
            "num_steps": num_steps,
            "latencies_s": latencies_s,
            "lat_mean_s": lat_mean_s,
            "is_chunked": is_chunked,
            "chunk_size": chunk_size,
        }

    @staticmethod
    def _build_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Builds the standard LIBERO observation dict from a raw env observation.

        Reuses the official image extraction (with the 180-degree rotation the OFT/OpenVLA
        training pipeline expects) and axis-angle proprioception conversion. Does NOT resize
        images -- that is model-specific and left to the wrapper (Adapter).
        """
        from experiments.robot.libero.libero_utils import (
            get_libero_image,
            get_libero_wrist_image,
            quat2axisangle,
        )

        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
        }
