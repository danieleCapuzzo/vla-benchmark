#!/usr/bin/env python
"""
CLI entrypoint: wires PolicyFactory -> wrapper -> LiberoEvaluator -> results_io.
Example:
    python run_eval.py --model oft --suite spatial --quant 8bit --rollouts 20
    python run_eval.py --model openvla --suite all --quant 8bit  # full benchmark
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

# set warning level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# stops tokenizer parallelism to avoid deadlock issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _suppress_noisy_warnings() -> None:
    """
    Silences cosmetic DeprecationWarnings/UserWarnings from torch/accelerate/importlib
    internals.
    """
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r"`torch\.jit\..*` is deprecated"
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"the load_module\(\) method is deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"The 'warn' method is deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`resume_download` is deprecated.*",
    )


_suppress_noisy_warnings()

import torch  # noqa: E402

# import repo-root packages
# ensure clean import regardless of caller cwd
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config.constants import ALL_SUITES, NUM_CAMERAS_BY_MODEL, RENDER_RESOLUTION_BY_MODEL  # noqa: E402
from config.loader import FROZEN_ROLLOUTS_PER_TASK, RunConfig, load_config  # noqa: E402
from core.factory import PolicyFactory  # noqa: E402
from core.evaluator import LiberoEvaluator  # noqa: E402

# lazy factory registration
_WRAPPER_MODULE_BY_MODEL = {
    "openvla": "wrappers.openvla_wrapper",
    "oft": "wrappers.oft_wrapper",
    "mail": "wrappers.mail_wrapper",
    "robomamba": "wrappers.robomamba_wrapper",
}
from results_io.results import (  # noqa: E402
    write_all_suites_csv,
    write_episodes_jsonl,
    write_summary_csv,
)

# kill warnings
_suppress_noisy_warnings()

# per-model default config file (config/config_<model>.json).
_DEFAULT_CONFIG_BY_MODEL = {
    "openvla": "config_openvla.json",
    "oft": "config_oft.json",
    "mail": "config_mail.json",
    "robomamba": "config_robomamba.json",
}

# camera count is fixed per model, not a runtime parameter.
NUM_CAMERAS_BY_MODEL = {
    "openvla": 1,
    "oft": 2,
    "mail": 2,
    "robomamba": 1,
}


def default_config_for(model: str) -> Path:
    """Returns the default `config/config_<model>.json` path for the given --model."""
    return REPO_ROOT / "config" / _DEFAULT_CONFIG_BY_MODEL[model]


def parse_args() -> argparse.Namespace:
    """
    Parses keyboard arguments
    """
    parser = argparse.ArgumentParser(description="VLA-on-LIBERO testbench CLI")
    parser.add_argument("--model", required=True, choices=["openvla", "oft", "mail", "robomamba"])
    parser.add_argument(
        "--suite",
        required=True,
        nargs="+",
        choices=["spatial", "object", "goal", "long", "all"],
        help="One or more suites, or 'all' for the full benchmark in one invocation.",
    )
    parser.add_argument("--quant", default=None, choices=["none", "8bit", "4bit"])
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Rollouts per task (protocol default: 20)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON (default: config/config_<model>.json, auto-selected by --model)",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save an MP4 replay per trial to <suite_dir>/videos/ (debug aid; off by "
        "default since it writes one file per rollout).",
    )
    parser.add_argument(
        "--language",
        action="store_true",
        help="MaIL only: use natural language conditioning to solve the tasks.",
    )
    return parser.parse_args()


def resolve_suites(requested: List[str]) -> List[str]:
    """
    Expands 'all' and de-duplicates while preserving first-seen order.
    """
    if "all" in requested:
        return list(ALL_SUITES)
    seen = []
    for s in requested:
        if s not in seen:
            seen.append(s)
    return seen


def _suite_dir_name(suite: str, language: bool) -> str:
    """Subfolder name for a suite's results within a run_dir (see `run_one_suite`)."""
    return f"{suite}_language" if language else suite


def _resolve_run_dir(
    output_dir_path: Path,
    model: str,
    quant: str,
    num_cameras: int,
    suite_dir_names: List[str],
) -> Path:
    """Builds `results/{model}_{quant}_{ncam}cam/`, reusing an existing directory of that
    name as long as none of `suite_dir_names` (the suites about to be run) already exist
    inside it -- so e.g. evaluating a new suite lands alongside existing ones in the same
    `mail_none_2cam/` instead of always forking into a `mail_none_2cam2/` sibling. Only
    appends a progressive numeric suffix to `model` (model2, model3, ...) once a candidate
    directory already has results for one of `suite_dir_names`, to avoid overwriting them.
    """
    # numbering starts at 2 (bare `model` is the first run, `model2` the first collision)
    suffix = ""
    n = 1
    while True:
        candidate = output_dir_path / f"{model}{suffix}_{quant}_{num_cameras}cam"
        if not candidate.exists():
            return candidate
        if not any((candidate / name).exists() for name in suite_dir_names):
            return candidate
        n += 1
        suffix = str(n)


def run_one_suite(
    *,
    model: str,
    suite: str,
    cfg: RunConfig,
    quant: str,
    rollouts: int,
    seed: int,
    run_dir: Path,
    save_videos: bool = False,
    language: bool = False,
) -> Dict[str, Any]:
    """
    Runs a full evaluation for a single suite and writes its per-run result files.

    Args:
        model: Model id passed to `PolicyFactory.create` (e.g. "oft").
        suite: Suite name key into `cfg.suites` (e.g. "spatial").
        cfg: Loaded run configuration.
        quant: Quantization mode ("none", "8bit", "4bit").
        rollouts: Rollouts per task for this run.
        seed: Random seed, recorded in the summary row.
        run_dir: Top-level directory for this invocation (`results/{model}_{quant}_{ncam}cam/`,
                 shared by every suite in this invocation; see `main`). This suite's
                 `episodes.jsonl`/`summary.csv` are written to a `{suite}/` subfolder within it.
        save_videos: If True, saves an MP4 replay per trial under
                     `<suite_dir>/videos/` (see `core.core_utils.write_rollout_video`).
        language: True means to use language conditioning for the tasks 
                  (works if the model is trained) with language conditioning.

    Returns:
        The aggregated summary row dict for this suite (as written to its `summary.csv`),
        reused by `main` to assemble the cross-run summary when multiple suites are run.

    Raises:
        ValueError: If `suite` is not present in `cfg.suites`.
    """
    if suite not in cfg.suites:
        raise ValueError(
            f"Suite '{suite}' not found in config. Valid suites: {sorted(cfg.suites)}"
        )
    suite_cfg = cfg.suites[suite]

    # reset peak-memory stats first so this suite's VRAM peak
    # isn't inflated by the previous suite's (now-freed) model.
    torch.cuda.reset_peak_memory_stats()

    num_cameras = NUM_CAMERAS_BY_MODEL[model]
    render_resolution = RENDER_RESOLUTION_BY_MODEL[model]
    print(
        f"\n\n=== Loading {model} / {suite} ({quant}, {num_cameras}cam, "
        f"render={render_resolution}px) -- {suite_cfg.checkpoint} ==="
    )
    load_t0 = time.perf_counter()
    model_kwargs: Dict[str, Any] = {}
    if model == "mail":
        model_kwargs["language"] = language
    wrapper = PolicyFactory.create(
        model,
        checkpoint=suite_cfg.checkpoint,
        unnorm_key=suite_cfg.unnorm_key,
        **model_kwargs,
    )
    wrapper.load(quant=quant)
    print(f"[{suite}] Loaded in {time.perf_counter() - load_t0:.1f}s")
    # transformers/accelerate/bitsandbytes internals triggered by wrapper.load() reset the
    # global warnings.filters list as a side effect -- reassert ours (see
    # `_suppress_noisy_warnings` docstring).
    _suppress_noisy_warnings()

    suite_dir = run_dir / _suite_dir_name(suite, language)

    evaluator = LiberoEvaluator(
        task_suite_name=suite_cfg.task_suite_name,
        num_steps_wait=cfg.num_steps_wait,
        video_dir=(suite_dir / "videos") if save_videos else None,
        render_resolution=render_resolution,
    )
    episodes = evaluator.run(wrapper, num_rollouts_per_task=rollouts)

    metadata = wrapper.get_metadata()
    vram_peak_gb = wrapper.get_vram_footprint_gb()

    write_episodes_jsonl(suite_dir / "episodes.jsonl", episodes)
    row = write_summary_csv(
        suite_dir / "summary.csv",
        episodes,
        metadata,
        seed=seed,
        vram_peak_gb=vram_peak_gb,
    )

    print(f"[{suite}] Run complete. Results written to {suite_dir}")
    print(f"[{suite}] VRAM peak: {vram_peak_gb:.2f} GB")

    # free the model before next suite
    # explicitly call garbage collection
    # clear cache and sync torch to avoid OOM errors
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return row


def main() -> None:
    args = parse_args()

    # trigger factory model registration (@PolicyFactory.register)
    importlib.import_module(_WRAPPER_MODULE_BY_MODEL[args.model])

    cfg_path = Path(args.config) if args.config else default_config_for(args.model)
    cfg = load_config(cfg_path)

    # default to configuration if not provided
    quant = args.quant if args.quant is not None else cfg.quant
    rollouts = args.rollouts if args.rollouts is not None else cfg.rollouts_per_task
    seed = args.seed if args.seed is not None else cfg.seed
    output_dir = args.output_dir if args.output_dir is not None else cfg.output_dir

    # create the output directory where we log the experiments
    # independent of caller cwd
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = REPO_ROOT / output_dir_path

    suites = resolve_suites(args.suite)
    suite_dir_names = [_suite_dir_name(suite, args.language) for suite in suites]

    # results written to results/{model}_{quant}_{ncam}cam/, reusing an existing directory
    # of that name unless it already has results for one of `suites` (collisions handled
    # with progressive numbering ie.: {model}<2,3....>).
    run_dir = _resolve_run_dir(
        output_dir_path, args.model, quant, NUM_CAMERAS_BY_MODEL[args.model], suite_dir_names
    )

    # warn of different num rollouts than expected
    if rollouts != FROZEN_ROLLOUTS_PER_TASK:
        print(
            f"WARNING: --rollouts {rollouts} deviates from the frozen protocol value "
            f"({FROZEN_ROLLOUTS_PER_TASK}/task, see doc/ARCHITECTURE.md). "
            "Use only for smoke tests, not for reported results."
        )

    rows = [
        run_one_suite(
            model=args.model,
            suite=suite,
            cfg=cfg,
            quant=quant,
            rollouts=rollouts,
            seed=seed,
            run_dir=run_dir,
            save_videos=args.save_videos,
            language=args.language,
        )
        for suite in suites
    ]

    if len(suites) > 1:
        all_suites_path = run_dir / "all_suites.csv"
        write_all_suites_csv(all_suites_path, rows)
        print(f"Cross-run summary ({len(suites)} suites) written to {all_suites_path}")


if __name__ == "__main__":
    main()
