"""
Utils for evaluating policies in LIBERO simulation environments.
"""

import math
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

####################
# LIBERO utilities #
####################


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    return to_upright(obs["agentview_image"])


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    return to_upright(obs["robot0_eye_in_hand_image"])


####################
# image utilities  #
####################

def to_upright(image: np.ndarray) -> np.ndarray:
    """
    Rotate the LIBERO frame 180 degrees (upside-down -> upright)
    Args:
        image: uint8 HWC frame in LIBERO's native orientation.
    Returns:
        uint8 HWC copy, rotated 180 degrees.
    """
    return np.ascontiguousarray(image[::-1, ::-1])


def resize_frame(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resizes a uint8 HWC frame to `size` x `size` with INTER_AREA.
    """
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)



###################
# robot utilities #
###################
def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# Gripper convention pipeline for a policy trained on raw LIBERO demo actions
# (LIBERO env/demo convention: -1 = open, +1 = close; RLDS convention: 0 = close, 1 = open):
#
#   model output (bounds-normalized)
#     -> denormalize_actions        undo [-1,1] bounds scaling, back to raw LIBERO units
#     -> env_gripper_to_rlds        raw LIBERO gripper {-1,+1} -> RLDS [0,1]
#     -> normalize_gripper_action   RLDS [0,1] -> [-1,+1], binarized
#     -> invert_gripper_action      flip sign to match env's -1=open/+1=close
#
# A policy that already outputs an RLDS-native gripper skips env_gripper_to_rlds and
# denormalize_actions but still goes through the last two steps.


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def env_gripper_to_rlds(action: np.ndarray) -> np.ndarray:
    """
    Converts a raw LIBERO-convention gripper command into the RLDS convention the
    evaluator expects, for policies trained directly on LIBERO demo actions.
    Args:
        action: Action array with the gripper command in the last dimension.
    Returns:
        A copy with the last dimension remapped (-1 -> 1.0 open, +1 -> 0.0 close).
    """
    converted = action.copy()
    converted[..., -1] = (1.0 - converted[..., -1]) / 2.0
    return converted


def normalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """
    Per-dimension bounds normalization of an action array into [-1, 1], same scheme as
    OpenVLA-OFT's BOUNDS_Q99 (see external/openvla-oft's `_unnormalize_actions`): the 1st/99th
    percentiles are used instead of true min/max so a handful of outlier demo actions don't
    compress the rest of the range.
    Works on both np.ndarray (eval, on CPU) and torch.Tensor (training, possibly on GPU) --
    uses the `.clip()` method rather than `np.clip()` so a CUDA tensor input isn't forced
    through a numpy round-trip.
    Args:
        actions: (..., action_dim) raw actions.
        q01: (action_dim,) per-dimension 1st percentile of the training actions.
        q99: (action_dim,) per-dimension 99th percentile of the training actions.
    Returns:
        Normalized actions of the same shape, clipped to [-1, 1] (values outside [q01, q99]
        saturate at the boundary).
    """
    normalized = 2 * (actions - q01) / (q99 - q01 + 1e-8) - 1
    return normalized.clip(-1.0, 1.0)


def denormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverts `normalize_actions`, mapping a model's [-1, 1] bounds-normalized prediction back
    into raw LIBERO action units.

    Must be applied before the gripper convention conversion (`env_gripper_to_rlds`), since
    that conversion assumes a raw-unit gripper in [-1, +1].
    """
    return 0.5 * (actions + 1) * (q99 - q01 + 1e-8) + q01


####################
# video utilities  #
####################

def write_rollout_video(
    video_dir: str | Path,
    frames: List[np.ndarray],
    task_id: int,
    trial: int,
    success: bool,
) -> Path:
    """Writes one MP4 debug replay for a single trial.

    Uses `imageio` directly (same library the official `save_rollout_video` helpers in
    `openvla/`/`openvla-oft` use) rather than calling those helpers, since they hardcode a
    relative `./rollouts/{DATE}` output path -- writing there would land inside the
    read-only vendor repos (wrappers `os.chdir()` into them during `load()`).

    Args:
        video_dir: destination directory (typically the suite's results subfolder).
        frames: one rendered frame per model query (the rotated `full_image` shown to the
            policy), in playback order.
        task_id: LIBERO task index within the suite, used in the filename.
        trial: rollout index for this task, used in the filename.
        success: whether the episode succeeded, encoded in the filename.

    Returns:
        The path of the written MP4 file.
    """
    import imageio

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    outcome = "success" if success else "fail"
    path = video_dir / f"task{task_id}_trial{trial}_{outcome}.mp4"

    writer = imageio.get_writer(str(path), fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return path
